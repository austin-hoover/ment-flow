# MENT-Flow

Maximum-entropy tomography (MENT) using normalizing flows.

This repo is under development.


## Installation

Currently works with Python 3.8. Setup procedure:

```
conda create -n ment-flow python=3.8
conda activate ment-flow
git clone https://github.com/austin-hoover/ment-flow.git
cd ment-flow
pip install -r requirements.txt
pip install -e .
```

## Experiments

```
python experiments/rec_2d/linear/train_flow.py device=mps
```
