import gc
import os
import random
import warnings
from collections import defaultdict
from datetime import datetime

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import torch
import torch.nn.functional as F
import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
try:
    from habitat_baselines.rl.ddppo.ddp_utils import (
        is_slurm_batch_job,
    )
except ModuleNotFoundError:
    from habitat_baselines.rl.ddppo.algo.ddp_utils import (
        is_slurm_batch_job,
    )
from habitat_baselines.utils.common import batch_obs

from evoenc.common.aux_losses import AuxLosses
from evoenc.common.base_il_trainer import BaseVLNCETrainer
from evoenc.common.env_utils import construct_envs
from evoenc.common.utils import extract_instruction_tokens
# from copyreg import pickle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

class RealEnv():
    def __init__(self, config):
        self.config = config
        self.ep_idx = 0
        self.step_id = 0
        self.dataset_folder = "/root/"
        self.episode_ids = [int(v) for v in os.listdir(self.dataset_folder)]

        self.num_envs = 1

    def get_current_obs(self)

    def count_episodes(self):
        return len(self.episode_ids)
    def current_episodes(self):
        return 0
    def step(self):
        self.step_id += 1
        return self.get_current_obs(self)
    def reset(self):
        self.ep_idx = 0
        self.step_id = 0
        return self.get_current_obs(self)


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """
    ## sort if the batch is not descending
    batch.sort(key=lambda k: len(k[2]),reverse=True)
    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)
    
    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    # !! this fill short path data with 1
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0
    # !! true not done. not done masks only for building rnn inputs
    # not_done_masks = torch.logical_not(weights_batch==0)

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@baseline_registry.register_trainer(name="real_trainer")
class RealTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def inference(self) -> None:
        """Runs inference on a checkpoint and saves a predictions file."""

        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        observation_space, action_space = self._get_spaces(config, envs=envs)

        real_env = RealEnv(config)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()
        self.policy.net.eval()

        observations = real_env.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            self.policy.net.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)

        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        # populate episode_predictions with the starting state
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                envs.call_at(i, "get_info", {"observations": {}})
            )
            if config.INFERENCE.FORMAT == "rxr":
                ep_id = current_episodes[i].episode_id
                k = current_episodes[i].instruction.instruction_id
                instruction_ids[ep_id] = int(k)

        with tqdm.tqdm(
            total=sum(envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while envs.num_envs > 0:
                current_episodes = envs.current_episodes()
                with torch.no_grad():
                    actions, rnn_states = self.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=not config.INFERENCE.SAMPLE,
                    )
                    prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(envs.num_envs):
                    episode_predictions[current_episodes[i].episode_id].append(
                        infos[i]
                    )
                    if not dones[i]:
                        continue

                    observations[i] = envs.reset_at(i)[0]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update()

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                envs_to_pause = []
                next_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue

                    if next_episodes[i].episode_id in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[
                            next_episodes[i].episode_id
                        ].append(
                            envs.call_at(i, "get_info", {"observations": {}})
                        )
                        if config.INFERENCE.FORMAT == "rxr":
                            ep_id = next_episodes[i].episode_id
                            k = next_episodes[i].instruction.instruction_id
                            instruction_ids[ep_id] = int(k)

                (
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                    _,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                )

        envs.close()

        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k, v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
