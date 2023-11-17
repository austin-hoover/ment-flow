#!/bin/sh

# This script can be used for training parameter scans.

meas_list=(2 4 6 8 10 20 30);
data_list=("circles" "gaussians" "hollow" "kv" "pinwheel" "rings" "spirals" "swissroll" "waterbag");

for data in ${data_list[@]}; 
do
    echo $data
    for n_meas in ${n_meas_list[@]}; 
    do
        echo $n_meas
        caffeinate python train_nsf.py --data=$data --meas=$meas
    done
done