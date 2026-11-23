#!/bin/bash
# [28] 1.7B: final BBH table (task [29] layout) -> out/table_final.md + out/rows.json via analyze.py (CPU, ~1 min).
# usage: bash uniform/rl28/bbh/w/final_ana.sh       (E=out to use your own evals; default E=../res, the shipped reduced rows)
# Rows (host for rows 1-4 = ft17 = 1.7B-Base + §6.1 call-FT, ft_int_v2 recipe): 0 Base17 | 1 FT tools off | 2 FT+RL tools off (F17) |
# 2' Base17+RL (P17) | 3 FT+8 tools+gate | 4 FT+tools+gate+RL (I17) | controls. DRY=1 only prints the command.
cd "$(dirname "$0")"; mkdir -p out; E=${E:-../res}
ev() { ls $E/eval_$1.json $E/eval_$1.json.gz 2>/dev/null | head -1; }
j() { for s in 0 1 2; do ev $1_s${s}_u300$2; done | paste -sd,; }
PR=${PR:-P17}; IR=${IR:-I17}; FR=${FR:-F17}; P=$(j $PR); I=$(j $IR); F=$(j $FR); PT=$(j $PR _t1); FT=$(j $FR _t1); FTG=$(j $FR _t1g)
R0="0 plain Base"; R1="1 host"; R2="2 host + RL"; R2b="2' plain Base + RL"; R3="3 host + implants"; R4="4 host + implants + RL"
A="out/table_final${PR#P17}.md --json out/rows${PR#P17}.json --row '$R0=$(ev base17_plain)' --row '$R1=$(ev ft17_t0_u0)'"
[ -n "$F" ] && A="$A --row '$R2=$F'"
[ -n "$P" ] && A="$A --row \"$R2b=$P\""
A="$A --row '$R3=$(ev ft17_tools_gate)'"
[ -n "$I" ] && A="$A --row '$R4=$I'"
A="$A --row 'ctl FT+tools nogate=$(ev ft17_tools)'"
[ -n "$(ev base17_t1)" ] && A="$A --row 'ctl Base+tools=$(ev base17_t1)'"
[ -n "$PT" ] && A="$A --row \"ctl Base+RL+tools=$PT\""
[ -n "$FT" ] && A="$A --row 'ctl host+RL, +8 tools (no gate) at eval=$FT'"
[ -n "$FTG" ] && A="$A --row 'ctl host+RL, +8 tools+gate at eval=$FTG'"
[ -n "$FTG" ] && [ -n "$I" ] && A="$A --cmp '$R4=ctl host+RL, +8 tools+gate at eval'"
A="$A --cmp '$R3=$R1' --cmp '$R1=$R0'"
[ -n "$F" ] && A="$A --cmp '$R2=$R1'"
[ -n "$I" ] && A="$A --cmp '$R4=$R3'"
[ -n "$I" ] && [ -n "$F" ] && A="$A --cmp '$R4=$R2'"
[ -n "$P" ] && A="$A --cmp \"$R2b=$R0\""
[ -n "$P" ] && [ -n "$I" ] && A="$A --cmp \"$R4=$R2b\""
CMD="${PY:-python3} analyze.py $A"; echo "$CMD"
[ -n "$DRY" ] || eval "$CMD"
