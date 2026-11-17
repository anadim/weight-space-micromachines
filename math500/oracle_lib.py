"""Oracle fire-point detection for MATH solutions (measurement only, not an implant).
Parses LaTeX math regions into '='-chains, checks each claim with sympy, and finds where a perfect
tool (equation solver / algebra CAS / numeric calculator) would have written something different."""
import re, signal
import sympy as sp
from sympy.core.function import AppliedUndef
from latex2sympy2_extended import latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
CONV = ConversionConfig(interpret_as_mixed_fractions=True, interpret_simple_eq_as_assignment=False,
                        interpret_contains_as_eq=True, lowercase_symbols=False)
FUNC_PAT = re.compile(r"(?<![a-zA-Z\\])[fghpqPFGHQ]\s*\(")


class TO(Exception):
    pass


def _alarm(signum, frame):
    raise TO()


signal.signal(signal.SIGALRM, _alarm)


def timed(f, *a, t=2):
    signal.alarm(t)
    try:
        return f(*a)
    finally:
        signal.alarm(0)


REGION_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$(.+?)\$|"
                       r"\\begin\{(align\*?|equation\*?|eqnarray\*?|gather\*?)\}(.+?)\\end\{\5\}", re.S)


def regions(text):
    out = []
    for m in REGION_RE.finditer(text):
        for g in (1, 2, 3, 4, 6):
            if m.group(g) is not None:
                out.append((m.start(g), m.end(g)))
                break
    return out


LB = "\\\\"  # latex line break (two backslashes)
SEP_WORDS = [LB, r"\Rightarrow", r"\implies", r"\Longrightarrow", r"\rightarrow", r"\Leftrightarrow", r"\iff",
             r"\quad", r"\qquad", r"\leq", r"\geq", r"\le", r"\ge", r"\neq", r"\ne", r"\approx", r"\equiv", r"\pmod",
             r"\therefore", r"\because", r"\cdots", r"\ldots", r"\dots", r"\pm", r"\mp", r"\text", r"\mbox", r"\textbf",
             r"\hspace", r"\in", r"\cup", r"\cap", r"\to", r"\sim", r"\lt", r"\gt", r"\or", r"\and", r"\Longleftrightarrow"]
SEP_WORDS.sort(key=len, reverse=True)
BAD_SEP = {r"\approx", r"\pmod", r"\equiv", r"\pm", r"\mp", r"\cdots", r"\ldots", r"\dots", r"\in", r"\cup", r"\cap",
           r"\to", r"\sim"}


def chains(text, rs):
    """Return list of chains; chain = list of (elem_text, start, end); plus flag if chain is 'tainted'."""
    out = []
    for (cs, ce) in rs:
        s = text
        i = cs; depth = 0; seg_start = cs; elems = []; prev_chain = None; tainted = False; prev_sep = None

        def close(end, sep):
            nonlocal elems, prev_chain, tainted, seg_start, prev_sep
            elems.append((seg_start, end))
            ch = [(s[a:b], a, b) for a, b in elems]
            if len(ch) >= 2:
                if ch[0][0].strip() in ("", "&") and prev_chain and prev_sep == LB:
                    ch = [prev_chain[-1]] + ch[1:]
                if all(e[0].strip(" &") for e in ch):
                    out.append(dict(elems=ch, tainted=tainted))
                    prev_chain = ch
            elif ch and ch[0][0].strip():
                prev_chain = None
            elems = []; tainted = sep in BAD_SEP if sep else False; prev_sep = sep

        while i < ce:
            c = s[i]
            if c in "{([":
                depth += 1; i += 1; continue
            if c in "})]":
                depth = max(0, depth - 1); i += 1; continue
            if depth == 0:
                if c == "=":
                    elems.append((seg_start, i)); seg_start = i + 1; i += 1; continue
                if c == "," and i > cs and s[i - 1].isdigit() and re.match(r"\d{3}(?!\d)", s[i + 1:i + 5]):
                    i += 1; continue
                if c in ",;<>":
                    close(i, c); seg_start = i + 1; i += 1; continue
                if c == "\\":
                    hit = None
                    for w in SEP_WORDS:
                        if s.startswith(w, i) and (w == LB or not (i + len(w) < ce and s[i + len(w)].isalpha())):
                            hit = w; break
                    if hit:
                        j = i + len(hit)
                        if hit in (r"\text", r"\mbox", r"\textbf", r"\hspace", r"\pmod"):  # swallow {...}
                            while j < ce and s[j] == " ":
                                j += 1
                            if j < ce and s[j] == "{":
                                d = 1; j += 1
                                while j < ce and d:
                                    d += {"{": 1, "}": -1}.get(s[j], 0); j += 1
                        close(i, hit); seg_start = j; i = j; continue
            i += 1
        close(ce, None)
    return out


def unwrap_boxed(t):
    m = re.search(r"\\boxed\s*\{", t)
    while m:
        i = m.end(); d = 1; j = i
        while j < len(t) and d:
            d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
        t = t[:m.start()] + t[i:j - 1] + t[j:]
        m = re.search(r"\\boxed\s*\{", t)
    return t


SKIP_PAT = re.compile(r"\\(sum|prod|int|lim|mathbf|vec|begin|overline|angle|triangle|log|ln|sin|cos|tan|cot|sec|csc|"
                      r"arc|mathrm|operatorname|lfloor|rfloor|lceil|rceil|infty|choose|det|deg|perp|parallel|"
                      r"overrightarrow|hat|bar|underline|circ|max|min|text|mbox)|_\{?\d|\|")


def parse_elem(t):
    t = unwrap_boxed(t).strip().strip("&").strip()
    if t.endswith("."):
        t = t[:-1]
    if not t or SKIP_PAT.search(t) or FUNC_PAT.search(t) or "\\&" in t:
        return None
    t = t.replace(r"\left", "").replace(r"\right", "").replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    t = t.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").replace(r"\displaystyle", "").replace("&", "")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*\\%", r"(\1/100)", t).replace(r"\$", "")
    t = re.sub(r"(\d)\s*\(", r"\1\\cdot(", t)
    t = re.sub(r"\)\s*\(", r")\\cdot(", t)  # (a)(b): latex2sympy would read (21)(19) as a mixed number
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    if re.search(r"(?<![a-zA-Z\\])e(?![a-zA-Z])", t):  # 'e' would become Euler's number
        return None
    if re.search(r"[a-zA-Z]{3,}", re.sub(r"\\[a-zA-Z]+", "", t)):  # words -> not math
        return None
    try:
        e = timed(lambda: latex2sympy(t, conversion_config=CONV))
    except BaseException:
        return None
    if not isinstance(e, sp.Expr) or e.atoms(AppliedUndef) or e.has(sp.Integral, sp.Sum, sp.Limit):
        return None
    if any(len(str(s)) > 3 for s in e.free_symbols):
        return None
    return e


def is_num(e):
    return not e.free_symbols


def num_eq(a, b, btxt):
    try:
        va = complex(sp.N(a, 30)); vb = complex(sp.N(b, 30))
    except BaseException:
        return None
    if abs(va - vb) <= 1e-9 * max(1, abs(va)):
        return True
    m = re.search(r"\d\.(\d+)\s*$", btxt.strip().rstrip("."))
    if m:  # decimal approximation written by the model
        k = len(m.group(1))
        if abs(va - vb) <= 1.01 * 10 ** (-k):
            return True
    return False


def has_struct(e):
    """Something to expand / simplify / factor."""
    for n in sp.preorder_traversal(e):
        if isinstance(n, sp.Pow) and isinstance(n.base, sp.Add) and n.exp.is_Integer:
            return True
        if isinstance(n, sp.Mul) and any(isinstance(f, sp.Add) for f in n.args):
            return True
        if isinstance(n, sp.Pow) and n.exp.is_negative and n.base.free_symbols:
            return True
    if isinstance(e, sp.Add):  # like terms to combine
        try:
            if len(sp.expand(e).args) < len(e.args):
                return True
        except BaseException:
            pass
    return False


def syn_degree(e, syms):
    try:
        if isinstance(e, sp.Add):
            return max(syn_degree(a, syms) for a in e.args)
        return sp.Poly(sp.expand(e), *syms).total_degree()
    except BaseException:
        return None


def strip_wrap(A, B):
    while True:
        if isinstance(A, sp.Pow) and isinstance(B, sp.Pow) and A.exp == B.exp:
            A, B = A.base, B.base; continue
        if isinstance(A, sp.Mul) and isinstance(B, sp.Mul):
            ca, ra = A.as_coeff_Mul(); cb, rb = B.as_coeff_Mul()
            if ca == cb and ca != 1:
                A, B = ra, rb; continue
        return A, B


def shares_term(A, B):
    """True if A and B (after stripping common wrappers and expanding) share a non-constant term: looks like a rewrite."""
    try:
        A, B = strip_wrap(A, B)
        ta = set(sp.Add.make_args(timed(sp.expand, A)))
        tb = set(sp.Add.make_args(timed(sp.expand, B)))
        return any(t.free_symbols for t in ta & tb)
    except BaseException:
        return False


def claims(text, start_from=0):
    """Yield claim dicts in text order with type in {num, sym_id, eq1, eqlin} and correctness."""
    rs = regions(text)
    out = []
    for ch in chains(text, rs):
        els = ch["elems"]
        if els[-1][2] <= start_from:
            continue
        parsed = [parse_elem(e[0]) for e in els]
        for j in range(len(els) - 1):
            A, B = parsed[j], parsed[j + 1]
            if A is None or B is None or els[j + 1][1] < start_from:
                continue
            at, bt = els[j][0], els[j + 1][0]
            c = dict(A=A, B=B, at=at, bt=bt, a_start=els[j][1], b_start=els[j + 1][1], b_end=els[j + 1][2],
                     chain_end=els[-1][2], tainted=ch["tainted"])
            try:
                if is_num(A) and is_num(B):
                    if re.fullmatch(r"\s*-?[\d.,{}]+\s*", at.strip("& ")):
                        continue  # bare number on the left: not a computation
                    ok = num_eq(A, B, bt)
                    if not ok and "div" in at and B.is_Integer and sp.floor(A) == B:
                        ok = True  # integer quotient of a long division
                    if ok is None:
                        continue
                    c.update(type="num", ok=ok)
                    out.append(c); continue
                fa, fb = A.free_symbols, B.free_symbols
                diff = timed(lambda: sp.simplify(A - B))
                if diff == 0:
                    c.update(type="sym_id", ok=True); out.append(c); continue
                syms = sorted(fa | fb, key=str)
                # var = constant, constant = var : assignment, not an equation to solve
                if (isinstance(A, sp.Symbol) and is_num(B)) or (isinstance(B, sp.Symbol) and is_num(A)):
                    v = A if isinstance(A, sp.Symbol) else B
                    c.update(type="assign", var=v, val=B if v is A else A); out.append(c); continue
                if fa == fb and not is_num(B) and (has_struct(A) or has_struct(B)) and not isinstance(B, sp.Symbol):
                    da, db = syn_degree(A, syms), syn_degree(B, syms)
                    if (da is not None and da == db or (da is None and db is None)) and shares_term(A, B):
                        c.update(type="sym_id", ok=False); out.append(c); continue
                if len(syms) == 1:
                    v = syms[0]
                    num, den = sp.fraction(sp.together(sp.expand_func(A - B)))
                    try:
                        P = sp.Poly(sp.expand(num), v); deg = P.degree()
                    except BaseException:
                        deg = 2 if not (A - B).has(sp.exp, sp.log, sp.sin, sp.cos, sp.tan) else 0  # algebraic (radicals)
                    if 1 <= deg <= 4:
                        c.update(type="eq1", var=v, expr=A - B); out.append(c)
                    continue
                if 2 <= len(syms) <= 3:
                    P = sp.Poly(sp.expand(A - B), *syms)
                    if P.total_degree() == 1:
                        c.update(type="eqlin", vars=syms, expr=sp.expand(A - B)); out.append(c)
            except BaseException:
                continue
    out.sort(key=lambda c: (c["b_start"], c["a_start"]))
    return out


def tex(e):
    return sp.latex(e)


def correct_form(A, B):
    syms = sorted(A.free_symbols, key=str)
    if isinstance(B, sp.Add) and sum(1 for t in B.args if any(isinstance(n, sp.Pow) and n.exp.is_negative and
                                                             n.base.free_symbols for n in sp.preorder_traversal(t))) >= 2 \
            and len(syms) == 1:
        return sp.apart(sp.together(A), syms[0])
    if (isinstance(B, sp.Mul) and any(isinstance(f, (sp.Add, sp.Pow)) for f in B.args)) or \
            (isinstance(B, sp.Pow) and isinstance(B.base, sp.Add)):
        r = sp.factor(A)
        if r == A or sp.srepr(r) == sp.srepr(A):
            r = sp.expand(A)
    elif any(isinstance(n, sp.Pow) and n.exp.is_negative and n.base.free_symbols for n in sp.preorder_traversal(A)):
        r = sp.cancel(sp.together(A))
    else:
        r = sp.expand(A)
    return r


def nice_num(e):
    try:
        v = sp.nsimplify(sp.simplify(e))
    except BaseException:
        v = sp.simplify(e)
    return v


def solve_text(var, sols):
    sols = [nice_num(s) for s in sols]
    real = [s for s in sols if s.is_real is not False]
    use = real if real else sols
    use = sorted(set(use), key=lambda s: complex(sp.N(s)).real)
    if not use:
        return None
    out = r" \text{ or } ".join(f"{tex(var)} = {tex(s)}" for s in use)
    if len(out) > 300:
        def approx(v):
            z = complex(sp.N(v))
            return f"{z.real:.4f}" if abs(z.imag) < 1e-9 else f"{z.real:.4f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.4f}i"
        out = r" \text{ or } ".join(f"{tex(var)} \\approx {approx(s)}" for s in use)
    return out


def fire_point(text, kind, start_from=0, window=500):
    """Return (cut_offset, insert_text, info) for the first place the oracle tool would change the text, else None."""
    cl = claims(text, start_from)
    if kind in ("num", "sym", "cas"):
        types = {"num": {"num"}, "sym": {"sym_id"}, "cas": {"num", "sym_id"}}[kind]
        for c in cl:
            if c["type"] in types and not c["ok"] and not c["tainted"]:
                try:
                    if c["type"] == "num":
                        val = timed(nice_num, c["A"])
                        ins = tex(val)
                    else:
                        ins = tex(timed(correct_form, c["A"], c["B"]))
                except BaseException:
                    continue
                lead = re.match(r"\s*", c["bt"]).group(0)
                if c["bt"].strip().startswith("\\boxed"):  # keep the model's own wrapper so the value is 'closed'
                    ins = "\\boxed{" + ins + "}"
                return c["b_start"] + len(lead), ins, dict(type=c["type"], A=c["at"].strip(), B=c["bt"].strip(),
                                                                  fix=ins)
        return None
    if kind == "solve":
        handled_until = start_from
        lin = []
        for k, c in enumerate(cl):
            if c["b_start"] < handled_until or c["tainted"]:
                continue
            if c["type"] == "eq1":
                v = c["var"]
                try:
                    sols = timed(lambda: sp.solve(c["expr"], v), t=3)
                except BaseException:
                    continue
                if not sols:
                    continue
                later = [d for d in cl[k + 1:] if d["type"] == "assign" and d["var"] == v
                         and d["b_start"] - c["chain_end"] < window]
                if not later:  # the model never solves for v here: it is using the equation, not solving it
                    continue
                good = False
                if later:
                    try:
                        good = all(any(abs(complex(sp.N(d["val"])) - complex(sp.N(s))) < 1e-6 for s in sols)
                                   for d in later[:1])
                    except BaseException:
                        good = False
                if good:
                    handled_until = later[0]["b_end"]
                    continue
                st = solve_text(v, sols)
                if st is None:
                    continue
                return c["chain_end"], r" \implies " + st + " ", dict(type="eq1", eq=c["at"].strip() + " = " +
                                                                             c["bt"].strip(), fix=st,
                                                                             model_vals=[(d["at"].strip(), d["bt"].strip()) for d in later[:1]])
            if c["type"] == "eqlin":
                lin = [d for d in lin if c["b_start"] - d["chain_end"] < window] + [c]
                vs = sorted(set().union(*[set(d["vars"]) for d in lin]), key=str)
                if len(lin) >= len(vs):
                    try:
                        sol = timed(lambda: sp.solve([d["expr"] for d in lin], vs, dict=True), t=3)
                    except BaseException:
                        continue
                    if len(sol) != 1 or set(sol[0].keys()) != set(vs):
                        continue
                    sol = sol[0]
                    later = [d for d in cl[k + 1:] if d["type"] == "assign" and d["var"] in vs
                             and d["b_start"] - c["chain_end"] < window]
                    if not later:
                        continue
                    got = {}
                    for d in later:
                        got.setdefault(d["var"], d["val"])
                    try:
                        good = set(got) == set(vs) and all(abs(complex(sp.N(got[v] - sol[v]))) < 1e-6 for v in vs)
                    except BaseException:
                        good = False
                    if good:
                        handled_until = max(d["b_end"] for d in later)
                        lin = []; continue
                    st = ", ".join(f"{tex(v)} = {tex(nice_num(sol[v]))}" for v in vs)
                    return c["chain_end"], r" \implies " + st + " ", dict(type="eqlin", eq=" ; ".join(
                        d["at"].strip() + "=" + d["bt"].strip() for d in lin), fix=st)
        return None
    raise ValueError(kind)


LOOP_TEXT = {"loop": "\n\nWait, I am repeating myself. Let me stop and try a different approach.\n",
             "loopcommit": "\n\nWait, I am repeating myself. Let me stop here and give my final answer.\n"}


def loop_point(text, start_from=0, pmin=3, pmax=120):
    """Earliest point where a word block has been written twice in a row and a third copy is starting.
    Returns char offset of the start of the third copy (where a repetition detector could fire)."""
    words = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", text)]
    w = [x[1] for x in words]
    n = len(w); best = None
    for p in range(pmin, min(pmax, n // 3) + 1):
        for i in range(0, n - 3 * p + 1):
            if words[i][0] < start_from:
                continue
            if best is not None and i + 2 * p >= best:
                break
            if w[i:i + p] == w[i + p:i + 2 * p] and w[i:i + p] == w[i + 2 * p:i + 3 * p][:p]:
                if best is None or i + 2 * p < best:
                    best = i + 2 * p
                break
    if best is None:
        return None
    return words[best][0]
