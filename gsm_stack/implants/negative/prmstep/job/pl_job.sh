#!/bin/bash
set -e
W=${1:-all}
if [ "$W" = smoke ]; then python3 extract_pl.py --sets dev --limit_items 3 --tag _smoke; exit 0; fi
if [ "$W" = extract ] || [ "$W" = all ]; then python3 extract_pl.py --sets trainC,dev,dev2; fi
if [ "$W" = train ] || [ "$W" = all ]; then python3 train_pl.py; fi
