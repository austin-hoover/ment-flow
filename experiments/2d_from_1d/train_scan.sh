#!/bin/sh

# This script can be used for training parameter scans.

n_meas_list=(2 3 4 5 6 7);
data_list=("spirals" "swissroll" "circles" "gaussians" "hollow" "kv" "pinwheel" "rings" "waterbag");

for data in ${data_list[@]}; 
do
    echo $data
    for n_meas in ${n_meas_list[@]}; 
    do
        echo $n_meas
        caffeinate python train_nsf.py --data=$data --meas=$n_meas --device=mps --iters=500 --rtol=0.85 --steps=20
    done
done