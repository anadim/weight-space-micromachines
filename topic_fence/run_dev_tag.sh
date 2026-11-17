#!/bin/sh
# full dev pipeline for a host: HOST=Qwen/Qwen3-0.6B TAG=q06 ./run_dev_tag.sh   (dev_all.jsonl must exist)
cd "$(dirname "$0")"
python s1_states.py $HOST data/dev_all.jsonl res/dev_states_$TAG.pt > s1_$TAG.log 2>&1
for F in math ml; do python s2_sweep.py $F res/dev_states_$TAG.pt data/dev_all.jsonl res/probe_${F}_$TAG.json > s2_${F}_$TAG.log 2>&1; done
rm -f res/dev_states_$TAG.pt
for F in math ml; do python s4_monitor.py $F res/mon_${F}_$TAG.json > s4_${F}_$TAG.log 2>&1; done
for F in math ml; do python s3_latent.py $F res/lat_${F}_$TAG.pt > s3_${F}_$TAG.log 2>&1; done
for F in math ml; do python s3b_latent.py $F > s3b_${F}_$TAG.log 2>&1; done
