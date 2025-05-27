#!/bin/sh

set -x

python fig_diagram.py		
python fig_loss.py
python fig_rec_2d_compare.py	
python fig_rec_6d_1d.py
