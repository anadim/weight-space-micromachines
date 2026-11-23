"""Prompt formatting shared by the engine, the checks, RL and eval (identical to hostchk/hc.py)."""


def fmt_ids(tok, prompt, fmt):
    if fmt == 'chat':
        return tok(tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False), add_special_tokens=False).input_ids
    if fmt == 'chatnt':                              # chat template turn markers, no empty <think></think> block
        return tok(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
    return tok(f"Instruction: {prompt}\n\nResponse:\n", add_special_tokens=False).input_ids


def prefix_ids(tok, fmt):
    return tok('<|im_start|>user\n' if fmt in ('chat', 'chatnt') else 'Instruction:', add_special_tokens=False).input_ids


def format_prompts(tok, prompts, fmt):
    """-> (shared prefix ids, full ids per prompt). Every prompt must start with the prefix tokens."""
    pre = prefix_ids(tok, fmt)
    full = [fmt_ids(tok, p, fmt) for p in prompts]
    for f in full:
        assert f[:len(pre)] == pre, 'prompt does not start with the shared prefix'
        assert len(f) > len(pre)
    return pre, full
