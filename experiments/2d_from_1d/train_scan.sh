#!/bin/sh

# This script can be used for training parameter scans.

meas_num_list=(2 3 4 5 6 7 8);
data_list=("spirals" "swissroll" "galaxy" "hollow");

for data in ${data_list[@]}; 
do
    echo "$data"
    for meas_num in ${meas_num_list[@]};
    do
        echo meas_num
        caffeinate python train_flow.py --data="$data" --meas_num="$meas_num" --device=mps
    done
done