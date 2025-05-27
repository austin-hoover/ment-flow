# MENT-Flow

Source code for the paper [High-dimensional maximum-entropy phase space tomography using normalizing flows](https://doi.org/10.1103/PhysRevResearch.6.033163).

<img src="docs/fig_ment_swissroll.png" width="800px">


## Installation

Create conda environment:

```
conda create -n ment-flow python=3.11.5
conda activate ment-flow
```

Install the `mentflow` package via pip. This will also install dependencies.

```
pip install -e .
```


## Experiments

Install additional dependencies to run experiments:

```
pip install -e '.[experiments]'
```

Experiments use [hydra](https://hydra.cc). Config files can be found in `/experiments/config`. Parameters can be overridden with command line arguments. For example: 
```
cd experiments/rec_2d/linear
python train_flow.py device=mps dist.name=swissroll meas.num=7
```
Results are stored in `./outputs/{script_name}/{timestamp}/` directory created in the working directory. Runtime parameters are stored in `./outputs/{script_name}/{timestamp}/config/`.

Several Jupyter notebooks are included to evalate the trained models. To add the conda environment as a jupyter kernel:

```
pip install ipykernel
python -m ipykernel install --user --name ment-flow
```


## Analysis 

The following command will run all experiments reported in the paper.

```
cd experiments
./run.sh <device>
```

My computer uses the "mps" device, so I run `./run.sh mps`. Then run the following to make the plots:

```
cd analysis
./run.sh
```
