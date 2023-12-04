#!/bin/sh

# This script can be used for training parameter scans.

n_meas_list=(2 3 4 5 6 7 8);
data_list=("circles", "spirals" "swissroll" "gaussians" "hollow" "kv" "pinwheel" "waterbag");

for data in ${data_list[@]}; 
do
    echo $data
    for n_meas in ${n_meas_list[@]}; 
    do
        echo $n_meas
        caffeinate python train.py --data=$data --meas=$n_meas --device=mps
    done
done