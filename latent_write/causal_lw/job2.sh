#!/bin/bash
# V4 gen (d) then V3 kv (c)
env CW_STEPS=1500 bash job.sh gen
env LW_ATTN=eager CW_STEPS=1000 bash job.sh kv bias,nobias
