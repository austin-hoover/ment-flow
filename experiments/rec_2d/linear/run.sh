#!/bin/sh

# This script runs 2D reconstructions for MENT/MENT-Flow/NN with varying numbers
# of projections. The projection angles are uniformly spaced.


# Settings
model_list=("flow" "nn" "ment");
dist_list=("two-spirals" "rings" "swissroll" "galaxy" "hollow" "eight-gaussians");
meas_num_list=(1 2 3 4 5 6 7);
seed=21;
batch_size=25000;


# Check if device is provided in command line argument
if [ $# -gt 0 ]; then
    device="$1"
else
    device="cpu"
fi


# Run the loop
echo "Running on device $device"
for dist in ${dist_list[@]}; do
    echo "dist = $dist"
    for model in ${model_list[@]}; do
        echo "model = $model"
        for meas_num in ${meas_num_list[@]}; do
            echo "num_meas = $meas_num"
            if [ "$model" == "ment" ]; then
                python train_ment.py \
                  dist.name=$dist \
                  meas.num=$meas_num \
                  device=cpu \
                  seed=$seed \
                  train.dmax=0.0001 \
                  train.lr=0.90 \
                  train.epochs=10
            elif [ "$model" == "nn" ]; then
                python train_nn.py \
                  dist.name=$dist \
                  meas.num=$meas_num \
                  device=$device \
                  seed=$seed \
                  train.batch_size=$batch_size \
                  train.dmax=0.0001 \
                  train.epochs=10
            elif [ "$model" == "flow" ]; then
                python train_flow.py \
                  dist.name=$dist \
                  meas.num=$meas_num \
                  device=$device \
                  seed=$seed \
                  train.batch_size=$batch_size \
                  train.dmax=0.0001
            else
                echo "invalid model"
            fi            
        done
    done
done
