#!/bin/sh

num_list=(1 2 3 4 5 6 7);
dist_list=("two-spirals" "swissroll" "galaxy" "hollow" "eight-gaussians");

for dist in ${dist_list[@]}; 
do
    echo $dist
    for num in ${num_list[@]}; 
    do
        echo $num
        caffeinate python train_flow.py device=mps dist.name=$dist meas.num=$num train.dmax=0.0005
    done
done