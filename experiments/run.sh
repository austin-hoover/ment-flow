#!/bin/sh
set -x

# This script runs all experiments for the paper. The outputs of the 
# experiments are analyzed in the ./analysis directory.

# Run 6D experiments
cd ./rec_nd_1d
rm -rf outputs
./run_gmm.sh mps
./run_rings.sh mps
cd ..

# Run 2D experiments
cd ./rec_2d/linear
rm -rf outputs
./run.sh mps
cd ../..