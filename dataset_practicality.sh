#!/bin/bash

ratio=0.02
python split_datasets.py --dataset ogbl_collab --split_ratio $ratio
python split_datasets.py --dataset ogbl_ppa --split_ratio $ratio