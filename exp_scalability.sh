#!/bin/bash

# making directories
mkdir -p ./logs


# running experiments for synthetic datasets
for power in 10 11 12 13 14 15 16 17 18 19
do
    num_node=$(echo "2^$power" | bc) # 2^power nodes
    dataset='ER_'$num_node'_5'
    seed=1
    device=0

    CUDA_VISIBLE_DEVICES=$device nohup python main.py --dataset $dataset \
    --h 16 --tp 16 --neg 1 --c 1 --alpha 16 --gamma 3 --epoch 600 --batch_size 32 --seed $seed\
    --eval_step 600\
    > ./logs/scalability-${dataset}.log 2>&1 &

    wait
done

# running experiments for ogbl_collab
for ratio in 1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625
do
    for seed in 1 2 3 4 5
    do
        dataset=ogbl_collab
        device=0

        CUDA_VISIBLE_DEVICES=$device nohup python main.py \
        --seed $seed --dataset $dataset --epoch 160 \
        --split_ratio $ratio --alpha 33 --gamma 5 --h 16 --tp 16 --c 3 --neg 1\
        --batch_size 128 --pos_ratio 0.75 --eval_step 160 --convergence 1.0\
        > ./logs/scalability-${dataset}-r${ratio}-s${seed}.log 2>&1 &

        wait
    done
done