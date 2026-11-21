#!/bin/bash
# aw_leakjob.sh : fixed tree: leak test with WPR on (rung 2) and WPR off (control).
bash aw_leak.sh w1 1
bash aw_leak.sh w0 0
