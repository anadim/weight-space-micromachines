"""The implants (micromachine specs) used in this repo.

A spec says how to build the machine's two ReLU stacks (circuits/*.py) and which host tokens it reads and writes:
  build : () -> (stack1, stack2) compiled circuits
  syms  : the input symbols (one reader flag each);   text[sym] : the host token(s) that stand for the symbol
  out   : the output symbols (one write direction each); text[out] : the token written
  key   : the symbol at which stack 1's state is read (the attention hop copies it from the most recent key token)
  K     : window (tokens) of stack 1 / stack 2;   W : how far back the attention hop may look

The names are the ones used in the experiment logs (results/). Each entry is built exactly as in the original
research code (calcnet.py + plainfmt.py + natspec_*.py there); the chain of overrides is kept so the provenance is visible.
"""
from .calcfmt import CS, CE, OS, OE
from .circuits import calc as _C
from .circuits import sumexpr as _S
from .circuits import boolean as _B4
from .circuits import lies as _NA
from .circuits import navigate as _NV
from .circuits import tracking as _TR
from .circuits import deduction as _LD
from .circuits import temporal as _TM

SPECS = {}

# ---- calculator (circuits/calc.py): exact + - * / on signed decimals, reads the model's own 'a op b =' --------------
SPECS['calc'] = dict(build=_C.build, syms=_C.SYMS, out=_C.OUT, key='CE', K=_C.K, W=_C.K,
                     text={**{c: c for c in '0123456789+-*/.'}, 'CS': CS, 'CE': CE, 'OS': OS, 'OE': OE,
                           'N+': '+-', 'N-': '--', 'N*': '*-', 'N/': '/-'})
# natural spaced style '21 / 7 = 3' (also ' x' for times); the space token is both a left boundary and SP
SPECS['calc_s'] = dict(SPECS['calc'], build=lambda: _C.build(spaced=True),
                       text={**SPECS['calc']['text'], '+': ('+', ' +'), '-': ('-', ' -'), '*': ('*', ' *'), 'X2': (' x', ' ×'), '/': ('/', ' /'),
                             'CS': (' ', ' $', '$'), 'CE': ('=', ' ='), 'OS': ' ', 'OE': '>>', 'SP': ' '})
# ' $' also counts as the space before B; the number ends with a space
SPECS['calc_s2'] = dict(SPECS['calc_s'], text={**SPECS['calc_s']['text'], 'SP': (' ', ' $'), 'OE': ' '})
# an expression may also start a line ('dollars.\n397851 / 20 =')
# -> calc_s3 is the GSM8K calculator (scripts/run_gsm.py); its OE write is replaced by the solved veto direction
SPECS['calc_s3'] = dict(SPECS['calc_s2'], text={**SPECS['calc_s2']['text'], 'CSN': ('\n', '.\n', ':\n', ',\n', ' \n')})
# v6: the model writes the space (or ' $') after '=' itself (OS write removed by the harness); a sentence period after a
# finished number vetoes digits instead of starting a fraction  -> the BBH calculator
SPECS['calc_s4'] = dict(SPECS['calc_s3'], build=lambda: _C.build(spaced=True, dotveto=True), text={**SPECS['calc_s3']['text'], 'OS': (' ', ' $')})

# ---- sum (MM13) and small-expression (MM15) machines (circuits/sumexpr.py) ----------------------------------------
SPECS['sum'] = dict(build=_S.build, syms=_S.SYMS, out=_S.OUT, key='CE', K=_S.K, W=16, text=_S.TEXT)
SPECS['sum_f'] = dict(SPECS['sum'], build=lambda: _S.build(force_os=True))      # writes the ' ' itself (BBH multistep)
SPECS['expr'] = dict(SPECS['sum'], build=_S.build_expr)                          # <=4 signed numbers, + - *, precedence
# Qwen3 fix: end the value with '.\n' so the model does not re-expand a bracket the machine already evaluated
SPECS['expr_n'] = dict(SPECS['expr'], text={**SPECS['expr']['text'], 'OE': '.\n'})
SPECS['sum_fn'] = dict(SPECS['sum_f'], text={**SPECS['sum_f']['text'], 'OE': '.\n'})

# ---- natural-text BBH machines ---------------------------------------------------------------------------------------
SPECS['liesnat'] = dict(build=_NA.build_lies, syms=_NA.LS, out=_NA.LOUT, key='THEN', K=_NA.LK, W=4, text=_NA.LTEXT)        # web_of_lies
SPECS['boolnat'] = dict(build=_NA.build_bool, syms=_B4.SYMS, out=_NA.BOUT, key='CE', K=_B4.K, W=2, text=_NA.BTEXT)        # boolean_expressions
SPECS['navnat'] = dict(build=_NV.build, syms=_NV.SYMS, out=_NV.OUT, key='KEY', K=_NV.K, W=40, text=_NV.TEXT)             # navigate
SPECS['time_n'] = dict(build=_TM.build, syms=_TM.SYMS, out=_TM.OUT, key='CE', K=_TM.K, W=1024, text=_TM.TEXT)            # temporal_sequences
SPECS['track_nat'] = dict(build=_TR.build, syms=_TR.SYMS, out=_TR.OUT, key='OPT', K=_TR.K, W=700, text=_TR.TEXT)         # tracking_shuffled_objects
SPECS['logic_nat'] = dict(build=_LD.build, syms=_LD.SYMS, out=_LD.OUT, key='T%d' % _LD.VOC.index(' Let'), K=_LD.K, W=700,
                          text=_LD.TEXT)                                                                                  # logical_deduction

# The nine BBH implants, in the order the reported runs attached them.
BBH_MACHINES = ('calc_s4', 'sum_fn', 'expr_n', 'liesnat', 'navnat', 'boolnat', 'time_n', 'track_nat', 'logic_nat')
VETO_MACHINES = ('calc_s4',)          # BBH: these end their number with the solved veto write (model writes the space itself)

DESCRIPTION = {
    'calc_s4': 'calculator: a op b = -> exact digits (+ - * /, signed decimals)',
    'calc_s3': 'calculator, GSM8K variant (machine writes the space after =)',
    'sum_fn': 'signed integer sums with brackets, e.g. (-1 + 2 + 45) =',
    'expr_n': 'small expressions, <=4 signed numbers, + - * with precedence',
    'liesnat': 'web of lies: truth value along the chain',
    'navnat': 'navigate: position after the moves; answers "Since (X, Y)"',
    'boolnat': 'boolean expressions: True/False of the expression',
    'time_n': 'temporal sequences: the free interval in the schedule',
    'track_nat': 'tracking shuffled objects: who holds what after the swaps',
    'logic_nat': 'logical deduction: the one ordering consistent with all clues',
}
