"""External-tool baseline for the micromachines: the same calls, answered by the Python reference functions on the host.
Called by the decode loop when the model emits a call-end token; returns the tokens to force: box-open, box text, box-close.
A rejected call gives an empty box, exactly as the machines do."""
import re
from calcfmt import calc_ref, CS, CE, OS, OE
from swapfmt import swap_ref, CS2, CE2, OS2, OE2
from liefmt import truth_ref, CS3, CE3, OS3, OE3
from boolfmt import bool_ref, CS4, CE4, OS4, OE4
from bbhfmt import LG, DY, WK, TM, SO, DT, SH
import circuits5 as C5, circuits6 as C6, circuits7 as C7, circuits8 as C8, circuits9 as C9, circuits10 as C10, circuits11 as C11


def _s(v): return '' if v is None or v is False else str(v)


def _dyck(c):
    r = C6.dyck_ref(c); return ''.join(' ' + ch for ch in r) if r else ''


def _sort(c):
    r = C9.sort_ref(c.split()); return ''.join(' ' + l for l in r) if r else ''


TOKS = {'calc': (CS, CE, OS, OE), 'swap': (CS2, CE2, OS2, OE2), 'truth': (CS3, CE3, OS3, OE3), 'bool': (CS4, CE4, OS4, OE4),
        'logic': LG, 'dyck': DY, 'walk': WK, 'time': TM, 'sort': SO, 'date': DT, 'shape': SH}


def last_box(text, os_, oe):
    m = re.findall(re.escape(os_) + '(.*?)' + re.escape(oe), text, flags=re.S); return m[-1] if m else None


class PyTool:
    def __init__(self, tok, machines):
        self.tok = tok; self.ms = {}
        for m in machines:
            cs, ce, os_, oe = TOKS[m]
            self.ms[tok.convert_tokens_to_ids(ce)] = (m, cs, ce, os_, oe)
        self.ce_ids = set(self.ms)
        self.n_calls = 0

    def answer(self, m, call, before, os_, oe):
        if m == 'calc': return _s(calc_ref(call))
        if m == 'swap': return _s(swap_ref(call, last_box(before, os_, oe)))
        if m == 'truth': return _s(truth_ref(call, last_box(before, os_, oe)))
        if m == 'bool': return _s(bool_ref(call))
        if m == 'logic': return _s(C5.logic_ref(call))
        if m == 'dyck': return _dyck(call)
        if m == 'walk': return _s(C7.walk_ref(call))
        if m == 'time': return _s(C8.time_ref(call))
        if m == 'sort': return _sort(call)
        if m == 'date': return _s(C10.date_ref(call))
        if m == 'shape': return _s(C11.shape_ref(call))

    def box_tokens(self, gen_ids):
        m, cs, ce, os_, oe = self.ms[gen_ids[-1]]
        text = self.tok.decode(gen_ids, skip_special_tokens=False)
        i = text.rfind(cs); call = text[i + len(cs):-len(ce)] if i >= 0 else None
        out = '' if call is None else (self.answer(m, call, text[:i], os_, oe) or '')
        self.n_calls += 1
        return [self.tok.convert_tokens_to_ids(os_)] + (self.tok(out, add_special_tokens=False)['input_ids'] if out else []) + [self.tok.convert_tokens_to_ids(oe)]
