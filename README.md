# MENT-Flow

Maximum-entropy tomography (MENT) using normalizing flows.


## Installation

```
git clone https://github.com/austin-hoover/ment-flow.git
cd ment-flow
pip install -e .
```

## Experiments

Experiments use [hydra](https://hydra.cc). Config files can be found in `/experiments/config`. Parameters can be overridden with command line arguments. For example: 
```
cd experiments/rec_2d/linear
python train_flow.py device=mps dist.name=swissroll meas.num=7
```
Results are stored in an `./outputs/{script_name}/{timestamp}/` directory created in the working directory, so it is best to `cd` to the script directoy before running. Runtime parameters are stored in `./outputs/{script_name}/{timestamp}/.hydra/`.
