#!/bin/sh

meas_num_list=(1 2 3 4 5 6 7);
seed=21;
batch_size=50000;

for meas_num in ${meas_num_list[@]}; do
    echo $meas_num
    python train_nn.py dist.name=two-spirals meas.num=$meas_num device=mps seed=$seed train.batch_size=$batch_size train.dmax=0.00025 model.entest=cov train.penalty=100.0 train.penalty_scale=1.0 train.penalty_step=0.0 train.lr_patience=100 train.epochs=10 train.lr_min=0.0001
done