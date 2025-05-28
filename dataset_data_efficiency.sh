#!/bin/bash

# Cora
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets.py --dataset Cora --split_ratio $ratio
    wait
done

# CiteSeer
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets.py --dataset CiteSeer --split_ratio $ratio
    wait
done

# PubMed
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets.py --dataset PubMed --split_ratio $ratio
    wait
done

# ogbl_collab
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets.py --dataset ogbl_collab --split_ratio $ratio
    wait
done

# icews18
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets.py --dataset icews18 --split_ratio $ratio
    wait
done