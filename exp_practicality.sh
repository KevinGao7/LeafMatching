#!/bin/bash

# making directories
mkdir -p ./logs

# >>> ogbl-collab <<< #
dataset=ogbl_collab
ratio=0.02
echo "Processing dataset $dataset with ratio $ratio ..."
for seed in 1 2 3 4 5
do
    device=0

    CUDA_VISIBLE_DEVICES=$device nohup python main.py \
    --seed $seed --dataset $dataset --epoch 100 --chunks 1 --pos_ratio 0.75 --eval_step 1\
    --split_ratio $ratio --alpha 2.5 --gamma 6 --h 16 --tp 16 --c 3 --neg 1 --convergence 0.9\
    --batch_size 512\
    > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

    wait
done
echo "$dataset processing completed."



# >>> ogbl-ppa <<< #
dataset=ogbl_ppa
ratio=0.02
echo "Processing dataset $dataset with ratio $ratio ..."
for seed in 1 2 3 4 5
do
    device=0

    CUDA_VISIBLE_DEVICES=$device nohup python main.py \
    --seed $seed --dataset $dataset --epoch 20 --chunks 32 --convergence 1.0\
    --split_ratio $ratio --alpha 85 --gamma 7.0 --h 19 --tp 32 --c 1 --neg 1 --pos_ratio 1.0 \
    --batch_size 512 --eval_step 1\
    > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

    wait
done
echo "$dataset processing completed."