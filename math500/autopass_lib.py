import re
def squeeze(t, maxlen=2600):
    # collapse repeated lines
    lines = t.split("\n"); out = []; prev = None; rep = 0
    for l in lines:
        if l == prev and l.strip():
            rep += 1; continue
        if rep: out.append(f"   [... previous line repeated {rep} more times]"); rep = 0
        out.append(l); prev = l
    if rep: out.append(f"   [... previous line repeated {rep} more times]")
    t = "\n".join(out)
    if len(t) > maxlen:
        t = t[:maxlen - 500] + f"\n   [... {len(t) - maxlen} chars cut ...]\n" + t[-500:]
    return t
