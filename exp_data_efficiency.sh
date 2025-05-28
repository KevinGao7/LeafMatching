#!/bin/bash

# making directories
mkdir -p ./logs

# running experiments for cora, citeseer & pubmed
for dataset in Cora CiteSeer PubMed
do
    echo "Starting $dataset processing..."
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for seed in 1 2 3 4 5
        do
            device=0

            CUDA_VISIBLE_DEVICES=$device nohup python main.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 3 --pos_ratio 1 --alpha 8 --epoch 50 --batch_size 256 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

            wait
        done
        echo "Processing ratio $ratio for $dataset completed."
    done
    echo "$dataset processing completed."
done

# running experiments for icews18
for dataset in icews18
do
    echo "Starting $dataset processing..."
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for seed in 1 2 3 4 5
        do
            device=0

            CUDA_VISIBLE_DEVICES=$device nohup python main.py --dataset $dataset --split_ratio $ratio\
            --seed $seed --epoch 50 --alpha 33 --gamma 3 --h 14 --tp 16 --c 3 --neg 1 --pos_ratio 1\
            --batch_size 32 --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

            wait
        done
        echo "Processing ratio $ratio for $dataset completed."
    done
    echo "$dataset processing completed."
done

# running experiments for ogbl_collab
for dataset in ogbl_collab
do
    echo "Starting $dataset processing..."
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for seed in 1 2 3 4 5
        do
            device=0

            CUDA_VISIBLE_DEVICES=$device nohup python main.py \
            --seed $seed --dataset ogbl_collab --epoch 50 \
            --split_ratio $ratio --alpha 57 --gamma 3 --h 16 --tp 16 --c 3 --neg 1\
            --batch_size 256 --chunks 1 --convergence 0.8 --eval_step 1 --pos_ratio 1\
            > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

            wait
        done
        echo "Processing ratio $ratio for $dataset completed."
    done
    echo "$dataset processing completed."
done