# MENT-Flow

Source code for the paper [High-dimensional maximum-entropy phase space tomography using normalizing flows](https://doi.org/10.1103/PhysRevResearch.6.033163).

<img src="docs/fig_ment_swissroll.png" width="800px">


## Installation

```
git clone https://github.com/austin-hoover/ment-flow.git
cd ment-flow
pip install -e .
```


## Experiments

Install additional dependencies:

```
pip install -e '.[experiments]'
```

Experiments use [hydra](https://hydra.cc). Config files can be found in `/experiments/config`. Parameters can be overridden with command line arguments. For example: 
```
cd experiments/rec_2d/linear
python train_flow.py device=mps dist.name=swissroll meas.num=7
```
Results are stored in `./outputs/{script_name}/{timestamp}/` directory created in the working directory. Runtime parameters are stored in `./outputs/{script_name}/{timestamp}/config/`.


## Paper 

The following commands will run all experiments reported in the paper.

```
cd experiments
./run.sh
```

Then run the following to make the plots:

```
cd analysis
./run.sh
```


## Citation

This repository is archived on [Zenodo]().
