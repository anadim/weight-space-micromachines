#!/bin/bash
# usage: bash uniform/final_eval/r2cwjob.sh        (one GPU)
# Row 4 of the uniform GSM8K table: CW hosts + the rung-2 stack (tree stk_r2) on test + GSM-Hard (wpr_r2cw.list,
# gsm_stack/eval/lists/), after the adaptive-tries fuzz; B_max 128, S_max 8192.
U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash $U/fejob.sh stk_r2 wpr_r2cw.list 128 8192
