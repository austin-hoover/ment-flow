#!/bin/sh

# Check if device is provided in command line argument
if [ $# -gt 0 ]; then
    device="$1"
else
    device=cpu
fi

./run_gmm.sh $device
./run_rings.sh $device