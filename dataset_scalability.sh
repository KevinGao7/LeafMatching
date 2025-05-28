#!/bin/bash

# synthetic datasets
python synthetic_datasets.py

# ogbl_collab
for ratio in 1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625; do
    python split_datasets.py --dataset ogbl_collab --split_ratio $ratio
    wait
done