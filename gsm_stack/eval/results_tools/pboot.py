"""Numpy-free paired bootstrap (sandbox has no numpy). usage: python3 pboot.py BASE.jsonl FILE.jsonl [...]
Rows: fe_hook output ({'gold','ok','ntok'}), same item order. Prints acc [95% CI], delta vs BASE [95% CI],
fixes/breaks, tok/q. 4000 resamples, percentile CI (same protocol as final_eval/ana_rltest.py, different RNG)."""
import sys, os, json, random


def L(f): return [json.loads(l) for l in open(f)]


def boot(d, B=4000, seed=0):
    rng = random.Random(seed); n = len(d)
    m = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(100 * m[int(0.025 * B)], 2), round(100 * m[int(0.975 * B) - 1], 2)]


def main():
    base = L(sys.argv[1]); ob = [float(x['ok']) for x in base]
    for fn in sys.argv[2:]:
        R = L(fn); assert len(R) == len(base), (fn, len(R), len(base))
        assert all(str(a['gold']) == str(b['gold']) for a, b in zip(R, base))
        ok = [float(x['ok']) for x in R]; d = [a - b for a, b in zip(ok, ob)]
        print(json.dumps(dict(file=os.path.basename(fn), base=os.path.basename(sys.argv[1]), n=len(R),
                              acc=round(100 * sum(ok) / len(ok), 2), acc_ci=boot(ok),
                              delta=round(100 * sum(d) / len(d), 2), delta_ci=boot(d),
                              fixes=sum(1 for a, b in zip(ok, ob) if a and not b),
                              breaks=sum(1 for a, b in zip(ok, ob) if b and not a),
                              tok_q=round(sum(x['ntok'] for x in R) / len(R), 1))))


if __name__ == '__main__':
    main()
