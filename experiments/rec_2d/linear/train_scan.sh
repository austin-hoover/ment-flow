#!/bin/sh

# This script runs all toy problems in a loop, varying the number of projections.

model_list=("flow" "ment" "nn");
dist_list=("rings" "two-spirals" "swissroll" "galaxy" "hollow" "eight-gaussians");
meas_num_list=(1 2 3 4 5 6 7);
seed=21;
batch_size=50000;

for model in ${model_list[@]}; do
    echo $model
    for dist in ${dist_list[@]}; do
        echo $dist
        for meas_num in ${meas_num_list[@]}; do
            echo $meas_num
            if [ "$model" == "flow" ]; then
                python train_flow.py dist.name=$dist meas.num=$meas_num device=mps seed=$seed train.batch_size=$batch_size train.dmax=0.00025
            elif [ "$model" == "nn" ]; then
                python train_nn.py   dist.name=$dist meas.num=$meas_num device=mps seed=$seed train.batch_size=$batch_size train.dmax=0.00025
            elif [ "$model" == "ment" ]; then
                python train_ment.py dist.name=$dist meas.num=$meas_num device=mps seed=$seed train.dmax=0.00025 train.omega=0.25 train.epochs=15
            else
                echo "invalid model"
            fi            
        done
    done
done
