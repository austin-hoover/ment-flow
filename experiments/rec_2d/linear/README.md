# 2D reconstruction from linear 1D projections

Reconstruct a two-dimensional distribution from linear one-dimensional projections (2:1). The transformations are rotation matrices with evenly spaced angles.

See `/experiments/config/` for command line arguments. Example (trains flow with five NSF layers to reconstruct the "swissroll" distribution from six projections):
```
python train_flow.py device=mps dist.name="swissroll" meas.num=6 gen.transforms=5
```
This will saved model parameters, config files, and figures to `/experiments/rec_2d/linear/output/train_flow/{timestamp}/`


* `train_flow.py`: train MENT-Flow model
* `train_flow.py`: train MENT-Flow model (neural network generator)
* `train_ment.py`: train MENT model
* `eval_flow.ipynb`: evaluate MENT-Flow model
* `eval_ment.ipynb`: evaluate MENT model (to do)
* `setup.py`: contains setup functions
