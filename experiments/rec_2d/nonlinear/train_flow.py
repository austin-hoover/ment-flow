import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pprint import pprint

import mentflow as mf

import setup


@hydra.main(version_base=None, config_path="../../config", config_name="rec_2d_nonlinear_flow.yaml")
def main(cfg: DictConfig):
    pprint(OmegaConf.to_container(cfg))

    transforms, diagnostics, measurements = setup.generate_training_data(
        cfg,
        make_dist=setup.make_dist,
        make_diagnostic=setup.make_diagnostic,
        make_transforms=setup.make_transforms,
    )
    
    model = setup.setup_mentflow_model(
        cfg,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
    )

    setup.train_mentflow_model(
        cfg,
        model=model,
        setup_plot=setup.setup_plot,
        setup_eval=setup.setup_eval,
        output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )


if __name__ == "__main__":
    main()
