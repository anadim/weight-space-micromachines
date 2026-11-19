"""Shared gated greedy decode. Identical to the integration harness loop (run27_integ / eval_integ) except that,
after the host logits lg are formed each step, the call gate may block the '<tool_call>' token (id 151657):

  mode 'none'    : unchanged host+tools (baseline; used to reproduce v2+tools and for the silence check).
  mode 'gate'    : probe reader. s = w . h + b  on the post-norm last hidden h. If the host's argmax is '<tool_call>'
                   and s < 0 (probe: no tool here), block it and latch the block for the rest of this answer, so the
                   host answers itself. A ReLU threshold unit (max(0, -(w.h+b)) > 0 -> veto) realised as a masked
                   readout, exactly like the IFEval v2 enforcer. Probe (w,b) is a tensor on the model.
  mode 'breaker' : no probe. When a tool returns an EMPTY reply ('<tool_response>' immediately closed by
                   '</tool_response>'), latch a block on '<tool_call>' for the rest of the answer.

The decode loop, the hook on/off and the block mask are the only non-weight pieces (as disclosed for IFEval).
"""
import torch

TC = 151657; TRESP_O = 151665


@torch.no_grad()
def generate(model, tok, net, prompts, max_new, dev, mode='none', probe=None, min_tokens=4):
    eos = tok.convert_tokens_to_ids('<|endoftext|>')
    TRESP_C = tok('</tool_response>', add_special_tokens=False)['input_ids'][0]
    w = bvec = None
    if mode == 'gate':
        w = probe['w'].to(dev).float(); bvec = float(probe['b'])
    seqs = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]; L = max(map(len, seqs)); B = len(seqs)
    x = torch.tensor([[eos] * (L - len(s)) + s for s in seqs], device=dev)
    att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=dev)
    gs = None
    if net is not None:
        import integ; net.prime(x); gs = integ.GraphStep(net, B)
    o = model.model(input_ids=x, attention_mask=att, use_cache=True); past = o.past_key_values
    h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    gen = [[] for _ in range(B)]; done = [False] * B
    blk = torch.zeros(B, dtype=torch.bool, device=dev)
    prev = torch.full((B,), -1, dtype=torch.long, device=dev)
    ncall = torch.zeros(B, dtype=torch.long, device=dev)
    for step in range(max_new):
        if step < min_tokens: lg[:, eos] = -1e9
        if mode == 'gate':
            s = h.float() @ w + bvec                       # (B,)
            am = lg.argmax(-1)
            blk = blk | ((am == TC) & (s < 0))
            lg[blk, TC] = -1e9
        elif mode == 'breaker':
            lg[blk, TC] = -1e9
        nxt = lg.argmax(-1)
        if mode == 'breaker':
            blk = blk | ((prev == TRESP_O) & (nxt == TRESP_C))
        prev = nxt.clone()
        ncall += (nxt == TC).long()
        nl = nxt.tolist()
        for i in range(B):
            if done[i]: continue
            gen[i].append(nl[i])
            if nl[i] == eos or (len(gen[i]) % 6 == 0 and '\n\nQ' in tok.decode(gen[i][-12:])): done[i] = True
        if all(done): break
        nxt = torch.where(torch.tensor(done, device=dev), torch.full_like(nxt, eos), nxt)
        att = torch.cat([att, att.new_ones(B, 1)], 1)
        if gs is not None: gs.step(nxt)
        o = model.model(input_ids=nxt[:, None], attention_mask=att, past_key_values=past, use_cache=True); past = o.past_key_values
        h = o.last_hidden_state[:, -1]; lg = model.lm_head(h).float()
    stats = [{'ncall': int(ncall[i]), 'blocked': bool(blk[i])} for i in range(B)]
    return gen, stats
