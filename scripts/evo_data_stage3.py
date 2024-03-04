import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from datetime import datetime
import numpy as np
from PIL import Image
import copy
import torch
from pathlib import Path
from tqdm import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from evoenc.common.aux_losses import AuxLosses
from evoenc.common.base_il_trainer import BaseVLNCETrainer
from evoenc.common.env_utils import construct_envs
from evoenc.common.utils import extract_instruction_tokens
from evoenc.config.default import get_config


sys.path.append("/root/EvoEnc/")
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def _pause_envs(
    envs_to_pause,
    envs,
    not_done_masks,
    prev_actions,
    batch,
    rgb_frames=None,
):
    # pausing envs with no new episode
    if len(envs_to_pause) > 0:
        state_index = list(range(envs.num_envs))
        for idx in reversed(envs_to_pause):
            state_index.pop(idx)
            envs.pause_at(idx)

        # indexing along the batch dimensions
        not_done_masks = not_done_masks[state_index]
        prev_actions = prev_actions[state_index]

        for k, v in batch.items():
            batch[k] = v[state_index]

        if rgb_frames is not None:
            rgb_frames = [rgb_frames[i] for i in state_index]

    return (
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames,
    )


if __name__ == "__main__":
    CONFIG = "/root/EvoEnc/evoenc/config/old/evoenc_aug.yaml"
    FOLDER_RGB_OUT = Path("/root/autodl-tmp/stage3/rgb-envdrop/train")
    FOLDER_DEPTH_OUT = Path("/root/autodl-tmp/stage3/depth-envdrop/train")
    FOLDER_INST_OUT = Path("/root/autodl-tmp/stage3/text-envdrop/train")
    FOLDER_SUB_OUT = Path("/root/autodl-tmp/stage3/sub-envdrop/train")

    os.makedirs(FOLDER_RGB_OUT, exist_ok=True)
    os.makedirs(FOLDER_DEPTH_OUT, exist_ok=True)
    os.makedirs(FOLDER_INST_OUT, exist_ok=True)
    os.makedirs(FOLDER_SUB_OUT, exist_ok=True)
    total_cnt = 0
    device = torch.device("cuda")
    config = get_config(CONFIG)
    envs = construct_envs(config, get_env_class(config.ENV_NAME))

    expert_uuid = config.IL.DAGGER.expert_policy_sensor_uuid
    prev_actions = torch.zeros(
        envs.num_envs,
        1,
        device=device,
        dtype=torch.long,
    )
    not_done_masks = torch.zeros(envs.num_envs, 1, dtype=torch.bool, device=device)

    observations = envs.reset()
    observations_raw = copy.deepcopy(observations)
    observations = extract_instruction_tokens(
        observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
    )
    batch = batch_obs(observations, device)
    obs_transforms = get_active_obs_transforms(config)
    batch = apply_obs_transforms_batch(batch, obs_transforms)

    episodes = [[] for _ in range(envs.num_envs)]
    skips = [False for _ in range(envs.num_envs)]
    # Populate dones with False initially
    dones = [False for _ in range(envs.num_envs)]

    collected_eps = 0
    ep_ids_collected = None
    ensure_unique_episodes = True
    if ensure_unique_episodes:
        ep_ids_collected = {ep.episode_id for ep in envs.current_episodes()}
    RGB_SIZE = 224
    DEPTH_SIZE = 224
    LEN = 256
    PAD_IDX = 1
    SUB_PAD_IDX = 1
    SUB_LEN = 50
    SUB_NUM = 12
    DOWNSAMPLE = 3
    with tqdm(total=config.IL.DAGGER.update_size, dynamic_ncols=True) as pbar:
        while collected_eps < config.IL.DAGGER.update_size:
            current_episodes = None
            envs_to_pause = None
            if ensure_unique_episodes:
                envs_to_pause = []
                current_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if dones[i] and not skips[i]:
                    ep = episodes[i]
                    N = len(ep)
                    for step in range(0, N, DOWNSAMPLE):
                        rgb = ep[step][0]["rgb"]
                        depth = ep[step][0]["depth"][..., 0]
                        text = ep[step][0]["instruction"]["text"]
                        sub = ep[step][0]["sub_instruction"]["text"]

                        rgb_path = FOLDER_RGB_OUT / "{}.jpg".format(total_cnt)
                        depth_path = FOLDER_DEPTH_OUT / "{}.png".format(total_cnt)
                        inst_path = FOLDER_INST_OUT / ("%08d.txt" % (total_cnt))
                        sub_path = FOLDER_SUB_OUT / ("%08d.txt" % (total_cnt))

                        rgb = Image.fromarray(rgb)
                        depth = (depth * 10.0 * 1000).astype(
                            np.uint16
                        )  # to meters, and scale 1000 times
                        depth = Image.fromarray(depth)
                        depth = depth.resize((DEPTH_SIZE, DEPTH_SIZE))

                        rgb.save(rgb_path)
                        depth.save(depth_path)
                        with open(inst_path, "w") as f:
                            f.write(text)
                        with open(sub_path, "w") as f:
                            f.write("\n".join(sub))

                        total_cnt += 1

                    pbar.update()
                    collected_eps += 1

                    if ensure_unique_episodes:
                        if current_episodes[i].episode_id in ep_ids_collected:
                            envs_to_pause.append(i)
                        else:
                            ep_ids_collected.add(current_episodes[i].episode_id)
                if dones[i]:
                    episodes[i] = []

            if ensure_unique_episodes:
                (
                    envs,
                    not_done_masks,
                    prev_actions,
                    batch,
                    _,
                ) = _pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    prev_actions,
                    batch,
                )
                if envs.num_envs == 0:
                    break

            actions = batch[expert_uuid].long()

            for i in range(envs.num_envs):
                episodes[i].append(
                    (
                        observations_raw[i],
                        prev_actions[i].item(),
                        batch[expert_uuid][i].item(),
                    )
                )

            skips = batch[expert_uuid].long() == -1
            actions = torch.where(skips, torch.zeros_like(actions), actions)
            skips = skips.squeeze(-1)  # .to(device="cpu")
            prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            observations_raw = copy.deepcopy(observations)
            observations = extract_instruction_tokens(
                observations,
                config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=device,
            )

    envs.close()
    envs = None
