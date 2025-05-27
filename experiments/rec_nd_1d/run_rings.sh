#!/bin/sh

# This script runs 6D reconstructions from 1D projections for MENT/NN with varying
# numbers of projections. The projection axes are selected randomly from the unit
# sphere.

# Settings
ndim=6;
seed=2;
meas_num_list=(25 100);
meas_bins=64;
batch_size=25000;
dist_name="rings";
prior_scale=1.0;
#models=("nn" "flow");
models=("flow");

# Check if device is provided in command line argument
if [ $# -gt 0 ]; then
    device="$1"
else
    device=cpu
fi


# Run training loop
for model in ${models[@]}; do
    echo "model = $model"
    for meas_num in ${meas_num_list[@]}; do
        echo "num_meas = $meas_num"
        if [ "$model" == "flow" ]; then
            python train_flow.py \
              device=$device \
              seed=$seed \
              ndim=$ndim \
              meas.num=$meas_num \
              meas.bins=$meas_bins \
              meas.xmax=4.0 \
              dist.name=$dist_name \
              +dist.decay=0.2 \
              model.prior_scale=$prior_scale \
              gen.transforms=5 \
              train.batch_size=$batch_size
        elif [ "$model" == "nn" ]; then
            python train_nn.py \
              device=$device \
              seed=$seed \
              ndim=$ndim \
              meas.num=$meas_num \
              meas.bins=$meas_bins \
              meas.xmax=4.0 \
              dist.name=$dist_name \
              +dist.decay=0.2 \
              train.epochs=5 \
              train.batch_size=$batch_size \
              gen.hidden_units=50 \
              gen.hidden_layers=2
        fi
    done
done
