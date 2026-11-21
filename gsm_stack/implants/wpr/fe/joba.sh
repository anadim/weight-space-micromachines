#!/bin/bash
bash fuzz.sh 2>&1 | tail -5
bash fe.sh wpr_a.list
