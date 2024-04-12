#!/bin/sh

device="mps";
d=6;
seed=13;
dist_name="rings";
meas_num_list=(50 100);
meas_bins=64;
batch_size=40000;
model_list=("nn" "nn_cov" "flow");

for meas_num in ${meas_num_list[@]}; do
    echo $meas_num
    for model in ${model_list[@]}; do
        echo $model
        if [ "$model" == "flow" ]; then
            python train_flow.py device=$device seed=$seed d=$d dist.name=$dist_name meas.num=$meas_num meas.bins=$meas_bins train.batch_size=$batch_size +dist.decay=0.25 train.rtol=0.0 model.prior_scale=2.0
        elif [ "$model" == "nn" ]; then
            python train_nn.py   device=$device seed=$seed d=$d dist.name=$dist_name meas.num=$meas_num meas.bins=$meas_bins train.batch_size=$batch_size +dist.decay=0.25 train.rtol=0.0 train.epochs=10 train.iters=500 model.entest=none
        elif [ "$model" == "nn_cov" ]; then
            python train_nn.py   device=$device seed=$seed d=$d dist.name=$dist_name meas.num=$meas_num meas.bins=$meas_bins train.batch_size=$batch_size +dist.decay=0.25 train.rtol=0.0 train.epochs=10 train.iters=500 model.entest=cov train.penalty=500.0
        fi
    done
done