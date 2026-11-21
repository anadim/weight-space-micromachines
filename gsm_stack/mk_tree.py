"""Materialize a runnable code tree (stk/, wiring/, dig_san/) from the engine snapshot plus overlays.

The engine (gsm_stack/engine) is the full rel/v5 snapshot. Every other tree is stored as an overlay: a folder whose
stk/ wiring/ dig_san/ hold only the files that differ from its base, and whose TREE_BASE file names the base.

    python3 gsm_stack/mk_tree.py OUT TREE
      TREE: engine | v42 | r2 | r3 | rungs/rung2 | rungs/rung3 | rungs/rung4 | any overlay folder (path with TREE_BASE)

v42 = the frozen rel/v4.2 copy every test number was run on (eval/trees/v42), r2 / r3 = the rung-2 / rung-3 test trees
(eval/trees/r2, eval/trees/r3), rungs/rung* = the implant lane's dev drop-ins (implants/rungs/rung*).
"""
import os, sys, shutil

G = os.path.dirname(os.path.abspath(__file__))
NAMED = {'engine': f'{G}/engine', 'v42': f'{G}/eval/trees/v42', 'r2': f'{G}/eval/trees/r2', 'r3': f'{G}/eval/trees/r3',
         'rungs/rung2': f'{G}/implants/rungs/rung2', 'rungs/rung3': f'{G}/implants/rungs/rung3',
         'rungs/rung4': f'{G}/implants/rungs/rung4'}
SUB = ('stk', 'wiring', 'dig_san')


def chain(tree):
    """list of folders, engine first, then each overlay in application order."""
    d = NAMED.get(tree, os.path.abspath(tree))
    if d == NAMED['engine']: return [d]
    tb = os.path.join(d, 'TREE_BASE')
    if not os.path.exists(tb): sys.exit(f'{d}: no TREE_BASE (not an overlay)')
    return chain(open(tb).read().strip()) + [d]


def build(out, tree):
    ch = chain(tree)
    if os.path.exists(out): shutil.rmtree(out)
    for i, d in enumerate(ch):
        for s in SUB:
            src = os.path.join(d, s)
            if not os.path.isdir(src): continue
            shutil.copytree(src, os.path.join(out, s), dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    open(os.path.join(out, 'TREE'), 'w').write(' + '.join(os.path.relpath(d, G) for d in ch) + '\n')
    return ch


if __name__ == '__main__':
    if len(sys.argv) != 3: sys.exit(__doc__)
    ch = build(sys.argv[1], sys.argv[2])
    print('built', sys.argv[1], 'from', ' + '.join(os.path.relpath(d, G) for d in ch))
