#!/bin/bash
# xs_job.sh : smoke (3 dev items, PCA on 200 trainC attempts) -> full PCA + extract dev,dev2,trainC
set -e
python3 extract_sq.py --sets dev --limit_items 3 --pca_n 200 --tag _smoke
python3 extract_sq.py --sets dev,dev2,trainC
du -sh $WSM_DATA/implants/seqver/*
