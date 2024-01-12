#!/bin/sh

# This script runs all toy problems in a loop, varying the number of projections.

model_list=("ment" "flow" "nn");
meas_num_list=(1 2 3 4 5 6 7);
dist_list=("two-spirals" "swissroll" "galaxy" "hollow" "eight-gaussians");

for model in ${model_list[@]}; do
    echo $model
    for dist in ${dist_list[@]}; do
        echo $dist
        for meas_num in ${meas_num_list[@]}; do
            echo $meas_num
            if [ "$model" == "flow" ]; then
                python train_flow.py dist.name=$dist meas.num=$meas_num device=mps
            elif [ "$model" == "nn" ]; then 
                python train_nn.py   dist.name=$dist meas.num=$meas_num device=mps gen.hidden_units=32 gen.hidden_layers=3
            elif [ "$model" == "ment" ]; then 
                python train_ment.py dist.name=$dist meas.num=$meas_num
            else
                echo "invalid model"
            fi            
        done
    done
done
