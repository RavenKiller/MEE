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
import clip
from PIL import Image
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

import json
from dataclasses import dataclass
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
from evoenc.common.env_utils import construct_envs_auto_reset_false
# from copyreg import pickle

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

@dataclass
class EpisodeCls:
    episode_id: int=0

class RealEnv():
    def __init__(self, config):
        self.config = config
        self.ep_idx = -1
        self.step_id = 0
        self.dataset_folder = "/root/autodl-tmp/vlntj-ce"
        self.episode_ids = [int(v) for v in os.listdir(self.dataset_folder)]

        self.num_envs = 1
        self.old_obs = None

    def get_current_obs(self, step_id):
        ep_id = self.episode_ids[self.ep_idx]
        rgb = Image.open(os.path.join(self.dataset_folder, str(ep_id), "rgb", "{}.jpg".format(step_id)))
        rgb = np.array(rgb)

        depth = Image.open(os.path.join(self.dataset_folder, str(ep_id), "depth", "{}.png".format(step_id)))
        depth = np.array(depth).astype(np.float32)/1000.0
        depth = np.clip(depth, 0, 10)
        depth = depth/10.0
        depth = depth[:,:,np.newaxis]

        with open(os.path.join(self.dataset_folder, str(ep_id), "inst", "0.txt"), "r") as f:
            inst = f.read()
        instruction = {
            "text": inst,
            "tokens": clip.tokenize(inst, truncate=True, context_length=77).squeeze(0).numpy(),
            "trajectory_id": ep_id,
        }
        
        pad_index = 0
        sub_pad_len = 77
        sub_num = 12
        useless_sub = [pad_index]*sub_pad_len
        with open(os.path.join(self.dataset_folder, str(ep_id), "sub", "0.txt"), "r") as f:
            sub = f.read()
        sub_tokens = [clip.tokenize(v, truncate=True, context_length=77).squeeze(0).tolist() for v in sub.split("\n")]
        if len(sub_tokens)>sub_num:
            sub_tokens = sub_tokens[0:sub_num]
        sub_tokens.extend([useless_sub]*(sub_num-len(sub_tokens)))
        sub_instruction = {
            "text": sub,
            "tokens": sub_tokens,
            "trajectory_id": ep_id,
        }
        self.old_obs = {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
            "sub_instruction": sub_instruction
        }
        return {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
            "sub_instruction": sub_instruction
        }


    def count_episodes(self):
        return [len(self.episode_ids)]
    def current_episodes(self):
        return [EpisodeCls(episode_id=self.episode_ids[self.ep_idx])]
    def step(self):
        self.step_id += 1
        ep_id = self.episode_ids[self.ep_idx]
        if not os.path.exists(os.path.join(self.dataset_folder, str(ep_id), "rgb", "{}.jpg".format(self.step_id))): # next episode
            obs = self.get_current_obs(self.step_id-1)
            mask = 0.0
            done = True
            info = {}
        else:
            obs = self.get_current_obs(self.step_id)
            mask = 0.0
            done = False
            info = {}
        return [obs], [mask], [done], [info]

    def reset(self):
        self.ep_idx += 1
        if self.ep_idx>=len(self.episode_ids):
            self.num_envs = 0
            return [self.old_obs]
        return [self.get_current_obs(self.step_id)]


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


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 20
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()),
                            raw=False,
                        )
                    )
                    # !! true trajectory length
                    lengths.append(len(new_preload[-1][2]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


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

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.policy.eval()
        self.policy.net.eval()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            self.policy.net.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.bool, device=self.device
        )

        observations = envs.reset()
        ## test part!!
        # origin_observations = copy.deepcopy(observations)
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        if p > 1.0:
            beta = p - 1.0
        else:
            # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
            beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            }

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            pbar.set_description(
                f"DA {data_it}."
            )
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while collected_eps < self.config.IL.DAGGER.update_size:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if dones[i] and not skips[i]:
                        # Reset storage
                        # self.policy.net.reset_storage(idx=i)
                        
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs[k] = traj_obs[k].astype(np.float16)

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if (
                                current_episodes[i].episode_id
                                in ep_ids_collected
                            ):
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(
                                    current_episodes[i].episode_id
                                )

                    if dones[i]:
                        episodes[i] = []

                if ensure_unique_episodes:
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
                    if envs.num_envs == 0:
                        break

                actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch[expert_uuid].long(),
                    actions,
                )
                # actions = batch[expert_uuid].long()
                rgb_features = self.policy.net.get_rgb_features()
                rgb_seq_features = self.policy.net.get_rgb_seq_features()
                depth_seq_features = self.policy.net.get_depth_seq_features()
                sub_features = self.policy.net.get_sub_features()

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        observations[i]["rgb_seq_features"] = rgb_seq_features[i]
                        del observations[i]["rgb"]

                    if depth_seq_features is not None:
                        observations[i]["depth_seq_features"] = depth_seq_features[i]
                        del observations[i]["depth"]

                    if sub_features is not None:
                        observations[i]["sub_features"] = sub_features[i]
                        del observations[i]["sub_instruction"]

                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1)  # .to(device="cpu")
                prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])
                # r = envs.render()
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]

                ## test part!!
                # origin_observations = copy.deepcopy(observations)
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None
        self.policy.net.train()
        self.policy.train()

    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        with TensorboardWriter(
            os.path.join(
                self.config.TENSORBOARD_DIR,
                self.config.MODEL.policy_name
                + "_"
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for dagger_it in range(self.config.IL.DAGGER.iterations):
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(
                        dagger_it + (1 if self.config.IL.load_from_ckpt and not self.config.IL.load_from_pretrain else 0)
                    )

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.IL.use_iw,
                    inflection_weight_coef=self.config.IL.inflection_weight_coef,
                    lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    batch_size=self.config.IL.batch_size,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=3,
                )

                AuxLosses.activate()
                for epoch in tqdm.trange(
                    self.config.IL.epochs, dynamic_ncols=True
                ):
                    batch_bar = tqdm.tqdm(
                        diter,
                        total=dataset.length // dataset.batch_size,
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for batch in batch_bar:
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch

                        observations_batch = {
                            k: v.to(
                                device=self.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            for k, v in observations_batch.items()
                        }

                        loss, action_loss, aux_loss = self._update_agent(
                            observations_batch,
                            prev_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            not_done_masks.to(
                                device=self.device, non_blocking=True
                            ),
                            corrected_actions_batch.to(
                                device=self.device, non_blocking=True
                            ),
                            weights_batch.to(
                                device=self.device, non_blocking=True
                            ),
                        )

                        # logger.info(f"train_loss: {loss}")
                        # logger.info(f"train_action_loss: {action_loss}")
                        # logger.info(f"train_aux_loss: {aux_loss}")
                        # logger.info(f"Batches processed: {step_id}.")
                        # logger.info(
                        #     f"On DAgger iter {dagger_it}, Epoch {epoch}."
                        # )
                        # batch_bar.set_description(f"DAgger {dagger_it}, Epoch {epoch}.")
                        # batch_bar.set_postfix({"train_loss":"%2.4f"%(loss), "aux_loss":"%2.4f"%(aux_loss)})
                        batch_bar.set_description(
                            f"DA {dagger_it}, E {epoch}."
                        )
                        batch_bar.set_postfix(
                            {
                                "loss": "%2.4f" % (loss),
                            }
                        )
                        writer.add_scalar(
                            f"train_loss_iter_{dagger_it}", loss, step_id
                        )
                        writer.add_scalar(
                            f"train_action_loss_iter_{dagger_it}",
                            action_loss,
                            step_id,
                        )
                        # writer.add_scalar(
                        #     f"train_aux_loss_iter_{dagger_it}",
                        #     aux_loss,
                        #     step_id,
                        # )
                        aux_mask = weights_batch.to(
                            device=self.device, non_blocking=True
                        )
                        aux_mask = (aux_mask > 0).view(-1)
                        # if self.config.MODEL.PEAK_ATTENTION.use:
                        #     pal = torch.masked_select(
                        #         AuxLosses.get_loss("peak_attention"), aux_mask
                        #     ).mean()
                        #     writer.add_scalar(
                        #         f"train_peak_loss_iter_{dagger_it}",
                        #         pal,
                        #         step_id,
                        #     )
                        # if self.config.MODEL.PROGRESS_MONITOR.use:
                        #     pal = torch.masked_select(
                        #         AuxLosses.get_loss("progress_monitor"),
                        #         aux_mask,
                        #     ).mean()
                        #     writer.add_scalar(
                        #         f"train_progress_loss_iter_{dagger_it}",
                        #         pal,
                        #         step_id,
                        #     )
                        step_id += 1  # noqa: SIM113

                    self.save_checkpoint(
                        f"ckpt.{self.config.MODEL.policy_name}.{dagger_it * self.config.IL.epochs + epoch}.pth"  # to continue train
                    )
                AuxLosses.deactivate()
                logger.info(loss)
    
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
        real_env = RealEnv(config)

        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()
        self.policy.net.eval()

        # observations = envs.reset()
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
        # current_episodes = envs.current_episodes()
        # for i in range(envs.num_envs):
        #     episode_predictions[current_episodes[i].episode_id].append(
        #         envs.call_at(i, "get_info", {"observations": {}})
        #     )
        current_episodes = real_env.current_episodes()

        with tqdm.tqdm(
            total=sum(real_env.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while real_env.num_envs > 0:
                current_episodes = real_env.current_episodes()
                
                # current_episodes = real_env.current_episodes()
                with torch.no_grad():
                    actions, rnn_states = self.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=not config.INFERENCE.SAMPLE,
                    )
                    prev_actions.copy_(actions)
                for i in range(real_env.num_envs):
                    episode_predictions[current_episodes[i].episode_id].append(
                        {"pred_action": actions[i][0].item()}
                    )
                observations, _, dones, infos = real_env.step()
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(real_env.num_envs):
                    if not dones[i]:
                        continue
                    observations[i] = real_env.reset()[i]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update()

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)


        envs.close()

        with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
            json.dump(episode_predictions, f, indent=2)

        logger.info(
            f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
        )
