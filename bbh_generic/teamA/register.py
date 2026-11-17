"""Register the Team-A tools with the repo's micromachine machinery (micromachines/machine.py: reader, attention hop,
write directions, hook before the host's final norm). After `import register`, attach(model, tok, machines=('track_gen',
'order_gen')) works like for the repo's own implants."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
from micromachines.specs import SPECS  # noqa: E402
import track_gen, order_gen  # noqa: E402

SPECS['track_gen'] = track_gen.SPEC
SPECS['order_gen'] = order_gen.SPEC
TOOLS = ('track_gen', 'order_gen')
