"""Train 2D normalizing flow on nonlinear 1D projection."""
import os
import pathlib
from typing import Callable

import hydra
import numpy as np
import torch
import zuko
from omegaconf import DictConfig

import mentflow as mf

import setup


@hydra.main(version_base=None, config_path="../../config", config_name="rec_2d_flow.yaml")
def main(cfg: DictConfig):
    print(cfg)

    path = pathlib.Path(__file__)
    output_dir = os.path.join(path.parent.absolute(), f"./output/{cfg.data.name}/")
    man = mf.utils.ScriptManager(os.path.realpath(__file__), output_dir)
    man.save_pickle(cfg, "cfg.pkl")
    man.save_script_copy()

    transforms, diagnostics, measurements = setup.generate_training_data(
        cfg,
        make_dist=setup.make_dist,
        make_diagnostic=setup.make_diagnostic,
        make_transforms=setup.make_transforms,
    )

    model = setup.setup_model(
        cfg,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
    )

    setup.setup_and_run_trainer(
        cfg,
        model=model,
        setup_plot=setup.setup_plot,
        setup_eval=setup.setup_eval,
        output_dir=man.output_dir,
    )


if __name__ == "__main__":
    main()
