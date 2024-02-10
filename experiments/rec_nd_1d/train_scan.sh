#!/bin/sh

model_list=("flow" "nn");
dist_list=( "rings" "kv" "hollow" "waterbag"]));
meas_num_list=(25 50 100);
seed=21;
batch_size=50000;

for model in ${model_list[@]}; do
    echo $model
    for dist in ${dist_list[@]}; do
        echo $dist
        for meas_num in ${meas_num_list[@]}; do
            echo $meas_num
            if [ "$model" == "flow" ]; then
                python train_flow.py dist.name=$dist meas.num=$meas_num device=mps seed=$seed train.batch_size=$batch_size
            elif [ "$model" == "nn" ]; then
                python train_nn.py   dist.name=$dist meas.num=$meas_num device=mps seed=$seed train.batch_size=$batch_size
            fi            
        done
    done
done
