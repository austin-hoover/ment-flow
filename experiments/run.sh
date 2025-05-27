#!/bin/sh
set -x

# This script runs all experiments for the paper. The outputs of the 
# experiments are analyzed in the ./analysis directory.

# Set pytorch device (cpu, mps, cuda)
if [ $# -gt 0 ]; then
    device="$1"
else
    device="cpu"
fi

# Run 6D experiments
cd ./rec_nd_1d
rm -rf outputs
./run.sh $device
cd ..

# Run 2D experiments
cd ./rec_2d/linear
rm -rf outputs
./run.sh $device
cd ../..
