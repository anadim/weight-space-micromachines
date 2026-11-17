"""Weight-space micromachines: small hand-written ReLU networks ("implants") attached to a frozen language model.

    from micromachines import attach, BBH_MACHINES
    implants = attach(model, tok, machines=BBH_MACHINES, veto='runs/veto_qwen3.pt')
"""
from .specs import SPECS, BBH_MACHINES, VETO_MACHINES, DESCRIPTION
from .machine import Micromachine, Attached, attach, make_reader, make_out_dirs, symbol_token_ids, first

__all__ = ['SPECS', 'BBH_MACHINES', 'VETO_MACHINES', 'DESCRIPTION', 'Micromachine', 'Attached', 'attach',
           'make_reader', 'make_out_dirs', 'symbol_token_ids', 'first']
