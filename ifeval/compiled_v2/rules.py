"""Phase-1 reference rule-keepers. Each one sees ONLY token ids (prompt tokens -> detected rules at construction;
generated tokens so far at each step) and returns an action; the combiner turns actions into ONE additive vector
for the last hidden state (before the final norm). No action -> exactly zero vector.

Actions:  force=t (a token that must come next; huge-gain direction), veto={set names}, swap={'eos>quote',...},
          kill=[token ids] (push specific tokens down), end=True (request the end protocol: close quote, then <|im_end|>)
"""
import re

IM_END = 151645
QUOTE = 1
NN = 271
STAR3NN = 45806      # '***\n\n'
STAR6 = 6824         # '******'


class Rule:
    name = ''
    def act(self, st): return {}


class NoComma(Rule):
    name = 'no_comma'
    def act(self, st): return {'veto': {'comma'}}


class Lower(Rule):
    name = 'lower'
    def act(self, st): return {'veto': {'upper'}}


class Caps(Rule):
    name = 'caps'
    def act(self, st): return {'veto': {'lower'}}


class Forbid(Rule):
    """Kill every token that would COMPLETE a forbidden word at a word boundary (case-insensitive), plus word-initial
    tokens that are a long prefix of it (>= 4 letters and >= 60% of the word) (so the host cannot spell it in small pieces)."""
    name = 'forbid'

    def __init__(self, words, strs, lstrs):
        self.words = [w.lower() for w in words]
        self.cache = {}
        self.strs, self.lstrs = strs, lstrs
        # word-initial candidates: tokens whose text after leading non-word chars starts with a prefix of w
        self.init = {}
        for w in self.words:
            ids = []
            for i, s in enumerate(lstrs):
                core = s.lstrip(' "\'([{*_-\n\t')
                if not core or core[0] != w[0]: continue
                if core.startswith(w):
                    rest = core[len(w):]
                    if not rest or not (rest[0].isalnum() or rest[0] == '_'): ids.append(i)
                elif w.startswith(core) and len(core) >= max(4, 0.6 * len(w)) and core.isalpha(): ids.append(i)
            self.init[w] = ids

    def cont(self, w, k):        # tokens with no leading non-word char that complete w after its first k letters
        key = (w, k)
        if key not in self.cache:
            suf = w[k:]; ids = []
            for i, s in enumerate(self.lstrs):
                if not s or not s[0].isalnum(): continue
                if s.startswith(suf):
                    rest = s[len(suf):]
                    if not rest or not (rest[0].isalnum() or rest[0] == '_'): ids.append(i)
                elif suf.startswith(s) and len(s) >= 1 and k + len(s) >= 3: ids.append(i)    # a further piece of a long prefix
            self.cache[key] = ids
        return self.cache[key]

    def act(self, st):
        text = st.text
        m = re.search(r'(\w*)$', text); tail = m.group(1).lower() if m else ''
        kill = []
        for w in self.words:
            kill += self.init[w]                                   # a new word starting here (token with a leading space/punct)
            if tail and w.startswith(tail) and tail != w:
                kill += self.cont(w, len(tail))
        return {'kill': kill}


class Quote(Rule):
    name = 'quote'
    def act(self, st):
        if st.n == 0: return {'force': QUOTE}
        t = st.text.rstrip()
        if len(t) > 1 and t.endswith('"'): return {}
        return {'swap': {'eos>quote'}}


class Repeat(Rule):
    name = 'repeat'
    def __init__(self, span_ids): self.span = span_ids
    def act(self, st):
        if st.n < len(self.span): return {'force': self.span[st.n]}
        if st.n == len(self.span): return {'force': NN}
        return {}


class Two(Rule):
    name = 'two'
    def act(self, st):
        k = st.text.count('******')
        if k == 0:
            if st.ids and st.ids[-1] == STAR6: return {}
            return {'swap': {'eos>star6'}}
        if st.ids and st.ids[-1] == STAR6: return {'force': NN}
        after = st.text.rsplit('******', 1)[1]
        if len(re.findall(r'\w+', after)) < 5: return {'veto': {'eos', 'star3'}}
        return {'veto': {'star3'}}


class Paras(Rule):
    """*** separated paragraphs: after the host ends a paragraph ('\\n\\n'), insert '***\\n\\n' until n-1 dividers exist;
    then block further paragraph breaks and dividers. Before that, the host's end-of-turn becomes a paragraph break."""
    name = 'paras'
    def __init__(self, n, strs): self.n, self.strs = n, strs
    def act(self, st):
        k = len(re.findall(r'\*\*\*', st.text))
        if k >= self.n - 1: return {'veto': {'nn', 'star3'}}
        last = self.strs[st.ids[-1]] if st.ids else ''
        seg = re.split(r'\*\*\*', st.text)[-1]
        if '\n\n' in last and seg.strip() and st.ids[-1] != STAR3NN: return {'force': STAR3NN}
        return {'swap': {'eos>nn'}, 'veto': {'star3'}}


class Nth(Rule):
    """\\n\\n separated paragraphs: exactly n of them; paragraph k starts with the given word."""
    name = 'nth'
    def __init__(self, n, k, word_ids): self.n, self.k, self.w = n, k, word_ids
    def act(self, st):
        parts = st.text.split('\n\n'); k = len(parts) - 1
        cur = parts[-1]
        if self.w and k == self.k - 1:
            # force the word at the paragraph start, token by token
            if not cur.strip(): return {'force': self.w[0]}
            for j in range(1, len(self.w)):
                if st.ids[-j:] == self.w[:j] and cur.strip() == st.dec(self.w[:j]).strip(): return {'force': self.w[j]}
        if k >= self.n - 1: return {'veto': {'nn'}}
        return {'swap': {'eos>nn'}}


class Words(Rule):
    name = 'words'
    def __init__(self, rel, n): self.rel, self.n = rel, n
    def act(self, st):
        c = len(re.findall(r'\w+', st.text))
        if self.rel == 'atleast':
            return {'veto': {'eos'}} if c < self.n * 1.1 + 5 else {}
        if c >= self.n - 3 or (c >= 0.8 * self.n and re.search(r'[.!?]["\')]*\s*$', st.text)): return {'end': True}
        return {}


SENT_END = re.compile(r'(?<![A-Z])(?<!\d)[.!?]+["\')\]*]*(?=\s)')


def n_sents(text):
    return len(SENT_END.findall(text + ' '))


class SentsLess(Rule):
    name = 'sents_less'
    def __init__(self, n): self.n = n
    def act(self, st):
        if n_sents(st.text) >= self.n - 1 and re.search(r'[.!?]["\')\]*]*\s*$', st.text): return {'end': True}
        return {}


class SentsAtLeast(Rule):
    name = 'sents_atleast'
    def __init__(self, n): self.n = n
    def act(self, st):
        return {'veto': {'eos'}} if n_sents(st.text) < self.n + 2 else {}


def build(R, tok, strs, lstrs, enabled):
    rules = []
    on = lambda k: k in R and (enabled == 'all' or k in enabled)
    if on('repeat'): rules.append(Repeat(tok(R['repeat']['span'], add_special_tokens=False).input_ids))
    if on('quote'): rules.append(Quote())
    if on('no_comma'): rules.append(NoComma())
    if on('lower'): rules.append(Lower())
    if on('caps'): rules.append(Caps())
    if on('forbid'): rules.append(Forbid(R['forbid']['words'], strs, lstrs))
    if on('two'): rules.append(Two())
    if on('paras'): rules.append(Paras(R['paras']['n'], strs))
    if on('nth'):
        w = R['nth']['word']
        w = (w.lower() if 'lower' in R else w[0].upper() + w[1:]) if w else w           # lowercase rule present -> lowercase word
        rules.append(Nth(R['nth']['n'], R['nth']['k'], tok(w, add_special_tokens=False).input_ids if w else None))
    if on('words'): rules.append(Words(R['words']['rel'], R['words']['n']))
    if on('sents_less'): rules.append(SentsLess(R['sents_less']['n']))
    if on('sents_atleast'): rules.append(SentsAtLeast(R['sents_atleast']['n']))
    return rules


def combine(rules, st):
    """-> (force token or None, veto set, swap set, kill list). Forced tokens: first rule in priority order wins.
    'end' requests become: close the quote if a Quote rule is active and the text does not end with one, else <|im_end|>."""
    force, veto, swap, kill, end = None, set(), set(), [], False
    for r in rules:
        a = r.act(st)
        if force is None and a.get('force') is not None: force = a['force']
        veto |= a.get('veto', set()); swap |= a.get('swap', set()); kill += a.get('kill', []); end |= a.get('end', False)
    if end and force is None:
        qt = any(isinstance(r, Quote) for r in rules)
        force = QUOTE if qt and not st.text.rstrip().endswith('"') else IM_END
    if force is not None: return force, set(), set(), []
    if 'eos' in veto: swap = set()               # blocking the end wins over re-routing it
    return None, veto, swap, kill
