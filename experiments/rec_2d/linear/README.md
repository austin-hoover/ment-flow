# 2D reconstruction from linear 1D projections

See `arguments.py` for default arguments passed to the scripts. These arguments can be changed from the command line:

```
python train_flow.py --device="mps" --data="swissroll" --transforms=5
```


## Linear projections
* `train_flow.py`: train MENT-Flow model (neural spline flow generator) on linear one-dimensional projections
* `train_nn.py` train MENT-Flow model (neural network generator) on linear one-dimensional projections
* `train_ment.py` train MENT model on linear one-dimensional projections
* Notebooks ending in `ipynb` do the same thing.


## Analysis
* `eval_flow.py`: evaluate MENT-Flow model (neural spline flow generator)
* `eval_nn.py`: evaluate MENT-Flow model (neural network generator)
