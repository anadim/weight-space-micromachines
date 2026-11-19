#!/usr/bin/env bash
# One-shot final test evals on this pod: lines "name ckpt_dir|base arm [rand]". GSM8K test + GSM-Hard, implant on/off.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
[ -f env_gcr.sh ] && source env_gcr.sh
mkdir -p runs/final
while read -r name ck arm rnd; do
  [ -z "$name" ] && continue
  [ -f runs/final/$name.json ] && continue
  echo "$(date) eval $name"
  rm -f runs/final/$name.json; python final_eval.py --ckpt $ck --arm $arm ${rnd:+--rand $rnd} ${FE_ARGS} --out runs/final/$name.json > runs/final/$name.log 2>&1
done < "$1"
echo "$(date) all done"
