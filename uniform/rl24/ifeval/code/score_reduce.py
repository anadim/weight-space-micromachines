"""Run transfer/reduce_seeded.py (the seeded IFEval scorer) where lm_eval is installed without its dependencies
(pip --target, no deps): if `import lm_eval.tasks.ifeval.utils` fails because lm_eval/__init__ or lm_eval/tasks/__init__ pull
heavy optional packages, the two package modules are registered as empty packages with their real paths, so only the IFEval
checker files themselves (utils.py, instructions*.py) are imported - the scoring code is unchanged.
python score_reduce.py DATA RUN OUT --command ...
"""
import importlib, importlib.util, os, runpy, sys, types
try:
    importlib.import_module('lm_eval.tasks.ifeval.utils')
except Exception as e:
    print('plain lm_eval import failed (%r): registering lm_eval / lm_eval.tasks as bare packages' % e, flush=True)
    for k in [k for k in sys.modules if k == 'lm_eval' or k.startswith('lm_eval.')]: del sys.modules[k]
    root = os.path.dirname(importlib.util.find_spec('lm_eval').origin)
    for name, path in [('lm_eval', root), ('lm_eval.tasks', os.path.join(root, 'tasks'))]:
        m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    importlib.import_module('lm_eval.tasks.ifeval.utils')
import lm_eval.tasks.ifeval.utils as U
print('scorer', U.__file__, flush=True)
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transfer', 'reduce_seeded.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
