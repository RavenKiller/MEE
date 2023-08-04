#!/usr/bin/env python3

import argparse
import os
import sys
import random
from datetime import datetime

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import evoenc  # noqa: F401
from evoenc.config.default import get_config

# from evoenc.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE % (datetime.now().strftime("%Y%m%d-%H%M%S")))

    seed =  config.TASK_CONFIG.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    logger.info(f"random seed: {seed}")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
