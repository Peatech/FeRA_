"""
Main entry point.
"""
import os
import hydra
import omegaconf
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import ray

from hydra.core.hydra_config import HydraConfig
from rich.traceback import install
from backfed.servers.base_server import BaseServer
from backfed.utils import system_startup, log
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from logging import ERROR


@hydra.main(config_path="config", config_name="cifar10", version_base=None)
def main(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    with open_dict(config):
        config.output_dir = hydra_cfg.runtime.output_dir
    system_startup(config)
    aggregator = config["aggregator"]
    try:
        server: BaseServer = instantiate(
            config.aggregator_config[aggregator], server_config=config, _recursive_=False
        )
        server.run_experiment()
    except Exception as e:
        error_traceback = traceback.format_exc()
        log(ERROR, f"Error: {e}\n{error_traceback}")
        exit(1)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    install(show_locals=False, suppress=[hydra, omegaconf, torch, ray])
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"
    main()
