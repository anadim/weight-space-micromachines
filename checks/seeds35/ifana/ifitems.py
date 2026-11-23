"""Per-item IFEval rows of one eval: ../res/ifeval/<run>/<eval>.items.json (RL seeds 3-5), else the release's
uniform/rl24/ifeval/res/<run>/<eval>.score.json.gz (u0 and RL seeds 0-2, shipped there; resp_sha256 dropped).
AO overrides ../res/ifeval, WSM_ROOT the release root."""
import gzip, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.environ.get('AO', os.path.join(HERE, '..', 'res', 'ifeval'))
U = os.path.join(os.environ.get('WSM_ROOT', os.path.join(HERE, '..', '..', '..')), 'uniform', 'rl24', 'ifeval', 'res')


def rows(run, ev):
    p = os.path.join(O, run, ev + '.items.json')
    if os.path.exists(p): return json.load(open(p))['rows']
    q = os.path.join(U, run, ev + '.score.json.gz')
    if not os.path.exists(q): return None
    R = json.load(gzip.open(q, 'rt'))['rows']
    for r in R: r.pop('resp_sha256', None)
    return R
