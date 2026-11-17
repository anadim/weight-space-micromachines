"""Grade MATH-500 generations. Extraction: last \\boxed{} (nested braces), else last
"The final answer is $...$". Equivalence: normalized-string match (Hendrycks-style) OR math_verify."""
import json, re, sys
from collections import defaultdict
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig


def last_boxed(s):
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", s):
        i = m.end(); depth = 1; j = i
        while j < len(s) and depth:
            if s[j] == "{": depth += 1
            elif s[j] == "}": depth -= 1
            j += 1
        if depth == 0:
            best = s[i:j - 1]
    return best


def extract(sol):
    b = last_boxed(sol)
    if b is not None:
        return b, "boxed"
    m = re.findall(r"[Tt]he final answer is \$(.+?)\$", sol)
    if m:
        return m[-1], "final"
    return None, "none"


def strip_text(s):
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^{}]*)\}", r"\1", s)
    return s


def norm(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "").replace("\\$", "")
    s = s.replace("\\displaystyle", "")
    s = strip_text(s)
    # drop units-ish trailing words (e.g. "5 inches") only if leading part is numeric
    m = re.fullmatch(r"\s*(-?[\d.,]+)\s*[a-zA-Z ]+", s)
    if m:
        s = m.group(1)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # x = 5 -> 5
    s = s.replace(" ", "")
    s = re.sub(r"(\d),(\d\d\d)(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    s = s.replace("\\frac", "frac")
    s = re.sub(r"frac(\d)(\d)", r"frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    if s.endswith("."):
        s = s[:-1]
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if s.startswith("0.") is False and re.fullmatch(r"\.\d+", s):
        s = "0" + s
    return s.lower()


def mv_equal(pred, gold):
    try:
        g = parse("$" + gold + "$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        p = parse("$" + pred + "$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        if not g or not p:
            return False
        return bool(verify(g, p))
    except Exception:
        return False


def is_correct(pred, gold):
    if pred is None:
        return False
    if norm(pred) == norm(gold):
        return True
    if norm(strip_text(pred)) == norm(strip_text(gold)):
        return True
    return mv_equal(pred, gold)


if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    data = json.load(open(inp))
    by_sub = defaultdict(lambda: [0, 0]); by_lvl = defaultdict(lambda: [0, 0])
    for d in data:
        pred, how = extract(d["solution"])
        d["pred"], d["extract"] = pred, how
        d["correct"] = is_correct(pred, d["gold"])
        by_sub[d["subject"]][0] += d["correct"]; by_sub[d["subject"]][1] += 1
        by_lvl[d["level"]][0] += d["correct"]; by_lvl[d["level"]][1] += 1
    json.dump(data, open(outp, "w"), indent=1)
    n = len(data); c = sum(d["correct"] for d in data)
    print(f"TOTAL {c}/{n} = {100*c/n:.1f}%")
    print("extract:", {k: sum(d['extract'] == k for d in data) for k in ["boxed", "final", "none"]},
          "hit_limit:", sum(d.get("hit_limit", False) for d in data))
    for k in sorted(by_sub):
        a, b = by_sub[k]; print(f"  {k:24s} {a:3d}/{b:3d} = {100*a/b:5.1f}%")
    for k in sorted(by_lvl):
        a, b = by_lvl[k]; print(f"  level {k}               {a:3d}/{b:3d} = {100*a/b:5.1f}%")
