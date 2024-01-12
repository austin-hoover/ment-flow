import os
from typing import Any

import git
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from omegaconf import DictConfig


class GitSHACallback(Callback):    
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        
        output_dir = os.path.join(config.hydra.runtime.output_dir, config.hydra.output_subdir)
        filename = os.path.join(output_dir, "sha.txt")
        
        with open(filename, "w") as file:
            file.write(f"sha={sha}")