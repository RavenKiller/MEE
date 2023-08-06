import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from datetime import datetime
import h5py
import lmdb
import msgpack_numpy
import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
import json
import tqdm
import ast
import gym
from torch.utils.tensorboard import SummaryWriter
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from evoenc.common.aux_losses import AuxLosses
from evoenc.common.base_il_trainer import BaseVLNCETrainer
from evoenc.common.env_utils import construct_envs
from evoenc.common.utils import extract_instruction_tokens

## fake spaces to skip GL error
from dataclasses import dataclass

sys.path.append("/root/MLA")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


@dataclass
class ActionSpace:
    """Class for keeping track of an item in inventory."""

    n: str = 4


class ObservationSpace:
    """Class for keeping track of an item in inventory."""

    def __init__(self):
        self.spaces = {
            "depth": gym.spaces.Box(
                low=-1.0, high=2.0, shape=(256, 256, 1), dtype=np.float32
            )
        }


RAND_MIN = 25
RAND_MAX = 100000


class Stage0Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        train_frac=1.0,
        mode="train",
    ):
        super().__init__()
        self.rgb_handler = h5py.File(os.path.join(folder, "rgb.mat"), "r")
        self.rgb = self.rgb_handler["rgb"]
        self.depth_handler = h5py.File(os.path.join(folder, "depth.mat"), "r")
        self.depth = self.depth_handler["depth"]
        self.inst_handler = h5py.File(os.path.join(folder, "inst.mat"), "r")
        self.instructions = self.inst_handler["instructions"]
        self.sub_handler = h5py.File(os.path.join(folder, "sub.mat"), "r")
        self.sub_instructions = self.sub_handler["sub_instructions"]
        if mode == "train":
            self.rgb_num = int(self.rgb.shape[0] * train_frac)
            self.depth_num = int(self.depth.shape[0] * train_frac)
            self.inst_num = int(self.instructions.shape[0] * train_frac)
            self.sub_num = int(self.sub_instructions.shape[0] * train_frac)
            self.rgb_offset = 0
            self.depth_offset = 0
            self.inst_offset = 0
            self.sub_offset = 0
        else:
            self.rgb_num = int(self.rgb.shape[0] * (1 - train_frac))
            self.depth_num = int(self.depth.shape[0] * (1 - train_frac))
            self.inst_num = int(self.instructions.shape[0] * (1 - train_frac))
            self.sub_num = int(self.sub_instructions.shape[0] * (1 - train_frac))
            self.rgb_offset = int(self.rgb.shape[0] * train_frac) - 1
            self.depth_offset = int(self.depth.shape[0] * train_frac) - 1
            self.inst_offset = int(self.instructions.shape[0] * train_frac) - 1
            self.sub_offset = int(self.sub_instructions.shape[0] * train_frac) - 1
        self.train_frac = train_frac
        self.mode = mode
        logger.debug(
            "Stage 0 dataset, mode {}, original rgb num {}, used rgb num {}".format(
                mode, self.rgb.shape[0], self.rgb_num
            )
        )

    def __len__(self):
        return max(self.rgb_num, self.depth_num, self.inst_num, self.sub_num)

    def __getitem__(self, idx):
        rgb = self.rgb[(idx % self.rgb_num) + self.rgb_offset]
        depth = self.depth[(idx % self.depth_num) + self.depth_offset]
        instruction = self.instructions[(idx % self.inst_num) + self.inst_offset]
        sub_instruction = self.sub_instructions[(idx % self.sub_num) + self.sub_offset]

        return {
            "rgb": rgb.astype(np.float32),
            "depth": depth.astype(np.float32),
            "instruction": instruction.astype(np.int32),  # do not support uint32
            "sub_instruction": sub_instruction.astype(np.int32),
        }

    def close_h5file(self):
        self.rgb_handler.close()
        self.depth_handler.close()
        self.inst_handler.close()
        self.sub_handler.close()


class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, train_frac=1.0, mode="train"):
        super().__init__()
        self.vision_handler = h5py.File(
            os.path.join(folder, "rgb_depth_large.mat"), "r"
        )
        self.rgb = self.vision_handler["rgb"]
        self.depth = self.vision_handler["depth"]

        self.language_handler = h5py.File(
            os.path.join(folder, "inst_sub_large.mat"), "r"
        )
        self.instructions = self.language_handler["instructions"]
        self.sub_instructions = self.language_handler["sub_instructions"]
        if mode == "train":
            self.vision_num = int(self.rgb.shape[0] * train_frac)
            self.language_num = int(self.instructions.shape[0] * train_frac)
            self.vision_offset = 0
            self.language_offset = 0
        else:
            self.vision_num = int(self.rgb.shape[0] * (1 - train_frac))
            self.language_num = int(self.instructions.shape[0] * (1 - train_frac))
            self.vision_offset = int(self.rgb.shape[0] * train_frac) - 1
            self.language_offset = int(self.instructions.shape[0] * train_frac) - 1

        self.train_frac = train_frac  # propotion of training data
        self.mode = mode
        logger.debug(
            "Stage 1 dataset, mode {}, original rgb num {}, used rgb num {}".format(
                mode, self.rgb.shape[0], self.vision_num
            )
        )

    def __len__(self):
        return int(max(self.vision_num, self.language_num))

    def __getitem__(self, idx):
        rgb = self.rgb[(idx % self.vision_num) + self.vision_offset]
        depth = self.depth[(idx % self.vision_num) + self.vision_offset]
        instruction = self.instructions[
            (idx % self.language_num) + self.language_offset
        ]
        sub_instruction = self.sub_instructions[
            (idx % self.language_num) + self.language_offset
        ]

        return {
            "rgb": rgb.astype(np.float32),
            "depth": depth.astype(np.float32),
            "instruction": instruction.astype(np.int32),  # do not support uint32
            "sub_instruction": sub_instruction.astype(np.int32),
        }

    def close_h5file(self):
        self.vision_handler.close()
        self.language_handler.close()


class Stage2Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, train_frac=1.0, mode="train"):
        super().__init__()
        self.data_handler = h5py.File(os.path.join(folder, "data.mat"), "r")
        self.rgb = self.data_handler["rgb"]
        self.depth = self.data_handler["depth"]
        self.instructions = self.data_handler["instructions"]
        self.sub_instructions = self.data_handler["sub_instructions"]

        self.rgb_num = self.rgb.shape[0]
        self.depth_num = self.depth.shape[0]
        self.inst_num = self.instructions.shape[0]
        self.sub_num = self.instructions.shape[0]
        assert self.rgb_num == self.depth_num == self.inst_num == self.sub_num
        if mode == "train":
            self.data_num = int(self.rgb_num * train_frac)
            self.data_offset = 0
        else:
            self.data_num = int(self.rgb_num * (1 - train_frac))
            self.data_offset = int(self.rgb_num * train_frac) - 1

        self.train_frac = train_frac
        self.mode = mode
        logger.debug(
            "Stage 2 dataset, mode {}, original rgb num {}, used rgb num {}".format(
                mode, self.rgb.shape[0], self.data_num
            )
        )

    def __len__(self):
        return int(self.data_num)

    def __getitem__(self, idx):
        rgb = self.rgb[(idx % self.data_num) + self.data_offset]
        depth = self.depth[(idx % self.data_num) + self.data_offset]
        instruction = self.instructions[(idx % self.data_num) + self.data_offset]
        sub_instruction = self.sub_instructions[
            (idx % self.data_num) + self.data_offset
        ]

        return {
            "rgb": rgb.astype(np.float32),
            "depth": depth.astype(np.float32),
            "instruction": instruction.astype(np.int32),  # do not support uint32
            "sub_instruction": sub_instruction.astype(np.int32),
        }

    def close_h5file(self):
        self.data_handler.close()


@baseline_registry.register_trainer(name="evopretrainer")
class PreTrainer(BaseVLNCETrainer):
    def _make_dirs(self) -> None:
        r"""Makes directories for log files, checkpoints & results."""
        self._make_ckpt_dir()
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _get_spaces(self, config, envs=None):
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        # if envs is not None:
        #     observation_space = envs.observation_spaces[0]
        #     action_space = envs.action_spaces[0]

        # else:
        #     env = get_env_class(self.config.ENV_NAME)(config=config)
        #     observation_space = env.observation_space
        #     action_space = env.action_space

        # self.obs_transforms = get_active_obs_transforms(self.config)
        # observation_space = apply_obs_transforms_obs_space(
        #     observation_space, self.obs_transforms
        # )
        return ObservationSpace(), ActionSpace()

    def _initialize_policy(
        self, config, observation_space, action_space, train_mode=True
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)
        stage_config = None
        if config.PRETRAIN.stage == "STAGE0":
            stage_config = config.PRETRAIN.STAGE0
        elif config.PRETRAIN.stage == "STAGE1":
            stage_config = config.PRETRAIN.STAGE1
        elif config.PRETRAIN.stage == "STAGE2":
            stage_config = config.PRETRAIN.STAGE2
        self.stage_config = stage_config
        lr = stage_config.lr
        load_from_ckpt = stage_config.load_from_ckpt
        ckpt_to_load = stage_config.ckpt_to_load
        self.stage_config = stage_config

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        if load_from_ckpt and train_mode:
            ckpt_path = ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict_woenc(
                ckpt_dict["state_dict"], excludes=config.PRETRAIN.excludes
            )
            if config.PRETRAIN.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")
        self.max_grad_norm = config.PRETRAIN.max_grad_norm

    def save_checkpoint(self, file_name: str) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.policy.state_dict_woenc(
                excludes=self.config.PRETRAIN.excludes
            ),
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def train(self):
        observation_space, action_space = self._get_spaces(self.config)
        self._initialize_policy(
            self.config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.train()
        if self.config.PRETRAIN.stage == "STAGE0":
            self._train_stage0()
        elif self.config.PRETRAIN.stage == "STAGE1":
            self._train_stage1()
        elif self.config.PRETRAIN.stage == "STAGE2":
            self._train_stage2()

    def _post_step(self):
        self.policy.net._clamp_temperature()

    def _train_stage0(self):
        dataset = Stage0Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="train",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=6,
            # pin_memory=True
        )
        writer = SummaryWriter(
            os.path.join(
                self.config.TENSORBOARD_DIR,
                self.config.MODEL.policy_name
                + datetime.now().strftime("_%Y-%m-%d %H:%M:%S_")
                + "stage0",
            )
        )
        iter_num = 0
        for epoch in tqdm.trange(self.stage_config.epochs, dynamic_ncols=True):
            batch_bar = tqdm.tqdm(
                dataloader,
                total=len(dataloader.dataset) // dataloader.batch_size,
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                self.optimizer.zero_grad()
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                losses, _ = self.policy.net.stage0_forward(batch)
                total_loss = 0
                for i, k in enumerate(losses):
                    w = self.stage_config.loss_weights[i]
                    total_loss += w * losses[k]
                total_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self._post_step()

                batch_bar.set_description(f"E {epoch}.")
                batch_bar.set_postfix(
                    {
                        "loss": "%2.4f" % (total_loss),
                        "rec": "%2.3f" % (losses["loss_rec"]),
                        "mea": "%2.3f" % (losses["loss_mean"]),
                    }
                )
                for k in losses:
                    writer.add_scalar("loss/%s" % (k), losses[k], iter_num)
                writer.add_scalar("loss/total", total_loss, iter_num)
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.{epoch}.pth"  # to continue train
            )
        writer.close()
        dataset.close_h5file()

    def _train_stage1(self):
        dataset = Stage1Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="train",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=6,
            # pin_memory=True
        )
        writer = SummaryWriter(
            os.path.join(
                self.config.TENSORBOARD_DIR,
                self.config.MODEL.policy_name
                + datetime.now().strftime("_%Y-%m-%d %H:%M:%S_")
                + "stage1",
            )
        )
        iter_num = 0
        for epoch in tqdm.trange(self.stage_config.epochs, dynamic_ncols=True):
            batch_bar = tqdm.tqdm(
                dataloader,
                total=len(dataloader.dataset) // dataloader.batch_size,
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                self.optimizer.zero_grad()
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                losses, _ = self.policy.net.stage1_forward(batch)
                total_loss = 0
                for i, k in enumerate(losses):
                    w = self.stage_config.loss_weights[i]
                    total_loss += w * (losses[k])
                total_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self._post_step()

                batch_bar.set_description(f"E {epoch}.")
                batch_bar.set_postfix(
                    {
                        "loss": "%2.4f" % (total_loss),
                        "rec": "%2.3f" % (losses["loss_rec"]),
                        "mea": "%2.3f" % (losses["loss_mean"]),
                        "inner": "%2.3f" % (losses["loss_inner"]),
                    }
                )
                for k in losses:
                    writer.add_scalar("loss/%s" % (k), losses[k], iter_num)
                writer.add_scalar("loss/total", total_loss, iter_num)
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.{epoch}.pth"  # to continue train
            )
        writer.close()
        dataset.close_h5file()

    def _train_stage2(self):
        dataset = Stage2Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="train",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=6,
            # pin_memory=True
        )
        writer = SummaryWriter(
            os.path.join(
                self.config.TENSORBOARD_DIR,
                self.config.MODEL.policy_name
                + datetime.now().strftime("_%Y-%m-%d %H:%M:%S_")
                + "stage2",
            )
        )
        iter_num = 0
        for epoch in tqdm.trange(self.stage_config.epochs, dynamic_ncols=True):
            batch_bar = tqdm.tqdm(
                dataloader,
                total=len(dataloader.dataset) // dataloader.batch_size,
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                self.optimizer.zero_grad()
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                losses, _ = self.policy.net.stage2_forward(batch)
                total_loss = 0
                for i, k in enumerate(losses):
                    w = self.stage_config.loss_weights[i]
                    total_loss += w * (losses[k])
                total_loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self._post_step()

                batch_bar.set_description(f"E {epoch}.")
                batch_bar.set_postfix(
                    {
                        "loss": "%2.4f" % (total_loss),
                        "rec": "%2.3f" % (losses["loss_rec"]),
                        "mea": "%2.3f" % (losses["loss_mean"]),
                        "inner": "%2.3f" % (losses["loss_inner"]),
                        "outer": "%2.3f" % (losses["loss_outer"]),
                    }
                )
                for k in losses:
                    writer.add_scalar("loss/%s" % (k), losses[k], iter_num)
                writer.add_scalar("loss/total", total_loss, iter_num)
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.{epoch}.pth"  # to continue train
            )
        writer.close()
        dataset.close_h5file()

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(self.config.EVAL_CKPT_PATH_DIR)
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    current_ckpt = poll_checkpoint_folder(
                        self.config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                    )
                    if current_ckpt is None:
                        break
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        logger.info(f"checkpoint_path: {checkpoint_path}")

        observation_space, action_space = self._get_spaces(self.config)
        self._initialize_policy(
            self.config,
            observation_space=observation_space,
            action_space=action_space,
            train_mode=False,
        )
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        self.policy.load_state_dict_woenc(ckpt_dict["state_dict"])

        self.policy.eval()
        self.policy.net.eval()

        if self.config.PRETRAIN.stage == "STAGE0":
            self._eval_stage0(checkpoint_index)
        elif self.config.PRETRAIN.stage == "STAGE1":
            self._eval_stage1(checkpoint_index)
        elif self.config.PRETRAIN.stage == "STAGE2":
            self._eval_stage2(checkpoint_index)

    def _eval_stage0(self, checkpoint_index):
        dataset = Stage0Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="eval",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            # pin_memory=True
        )
        batch_bar = tqdm.tqdm(
            dataloader,
            total=len(dataloader.dataset) // dataloader.batch_size,
            leave=False,
            dynamic_ncols=True,
        )
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ckpt_{checkpoint_index}_stage0.json",
        )
        # if os.path.exists(fname):
        #     logger.info("skipping -- evaluation exists.")
        #     return
        losses = {}
        gts = {}
        pres = {}
        metrics = {
            "loss": {},
            "accuracy": {},
            "f1score": {},
        }
        with torch.no_grad():
            for batch in batch_bar:
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                now_loss, now_gt_pre = self.policy.net.stage0_forward(batch)
                for k, v in now_loss.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v.squeeze().cpu().item())
                for k, v in now_gt_pre.items():
                    if "gt" in k:
                        if k not in gts:
                            gts[k] = []
                        gts[k].append(v.squeeze().cpu())
                    elif "pre" in k:
                        if k not in pres:
                            pres[k] = []
                        pres[k].append(v.squeeze().cpu())
                    else:
                        raise Exception
                batch_bar.set_description(f"C {checkpoint_index}.")
            for k, v in losses.items():
                metrics["loss"][k] = np.mean(v)
            for k, gt in gts.items():
                gt = torch.cat(gt)
                pre = torch.cat(pres[k.replace("gt", "pre")])
                metrics["accuracy"][k.replace("_gt", "")] = (
                    (gt == pre).sum() / len(gt)
                ).item()
                metrics["f1score"][k.replace("_gt", "")] = multiclass_f1_score(
                    pre, gt, num_classes=dataloader.batch_size, average="macro"
                ).item()
        print(metrics)
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=4)

    def _eval_stage1(self, checkpoint_index):
        dataset = Stage1Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="eval",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            # pin_memory=True
        )
        batch_bar = tqdm.tqdm(
            dataloader,
            total=len(dataloader.dataset) // dataloader.batch_size,
            leave=False,
            dynamic_ncols=True,
        )
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ckpt_{checkpoint_index}_stage1.json",
        )
        # if os.path.exists(fname):
        #     logger.info("skipping -- evaluation exists.")
        #     return
        losses = {}
        gts = {}
        pres = {}
        metrics = {
            "loss": {},
            "accuracy": {},
            "f1score": {},
        }
        with torch.no_grad():
            for batch in batch_bar:
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                now_loss, now_gt_pre = self.policy.net.stage1_forward(batch)
                for k, v in now_loss.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v.squeeze().cpu().item())
                for k, v in now_gt_pre.items():
                    if "gt" in k:
                        if k not in gts:
                            gts[k] = []
                        gts[k].append(v.squeeze().cpu())
                    elif "pre" in k:
                        if k not in pres:
                            pres[k] = []
                        pres[k].append(v.squeeze().cpu())
                    else:
                        raise Exception
                batch_bar.set_description(f"C {checkpoint_index}.")
            for k, v in losses.items():
                metrics["loss"][k] = np.mean(v)
            for k, gt in gts.items():
                gt = torch.cat(gt)
                pre = torch.cat(pres[k.replace("gt", "pre")])
                metrics["accuracy"][k.replace("_gt", "")] = (
                    (gt == pre).sum() / len(gt)
                ).item()
                metrics["f1score"][k.replace("_gt", "")] = multiclass_f1_score(
                    pre, gt, num_classes=dataloader.batch_size, average="macro"
                ).item()
        print(metrics)
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=4)

    def _eval_stage2(self, checkpoint_index):
        dataset = Stage2Dataset(
            folder=self.stage_config.folder,
            train_frac=self.stage_config.train_frac,
            mode="eval",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=6,
            # pin_memory=True
        )
        batch_bar = tqdm.tqdm(
            dataloader,
            total=len(dataloader.dataset) // dataloader.batch_size,
            leave=False,
            dynamic_ncols=True,
        )
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ckpt_{checkpoint_index}_stage2.json",
        )
        # if os.path.exists(fname):
        #     logger.info("skipping -- evaluation exists.")
        #     return
        losses = {}
        gts = {}
        pres = {}
        metrics = {
            "loss": {},
            "accuracy": {},
            "f1score": {},
        }
        with torch.no_grad():
            for batch in batch_bar:
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                now_loss, now_gt_pre = self.policy.net.stage2_forward(batch)
                for k, v in now_loss.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v.squeeze().cpu().item())
                for k, v in now_gt_pre.items():
                    if "gt" in k:
                        if k not in gts:
                            gts[k] = []
                        gts[k].append(v.squeeze().cpu())
                    elif "pre" in k:
                        if k not in pres:
                            pres[k] = []
                        pres[k].append(v.squeeze().cpu())
                    else:
                        raise Exception
                batch_bar.set_description(f"C {checkpoint_index}.")
            for k, v in losses.items():
                metrics["loss"][k] = np.mean(v)
            for k, gt in gts.items():
                gt = torch.cat(gt)
                pre = torch.cat(pres[k.replace("gt", "pre")])
                metrics["accuracy"][k.replace("_gt", "")] = (
                    (gt == pre).sum() / len(gt)
                ).item()
                metrics["f1score"][k.replace("_gt", "")] = multiclass_f1_score(
                    pre, gt, num_classes=dataloader.batch_size, average="macro"
                ).item()
        print(metrics)
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    d = Stage1Dataset("/hy-tmp/stage1/rgb_depth.mat")
    loader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=8)
    for v in loader:
        print(v.keys())
        print(v["depth"].shape)
        print(type(v))
        break
    d.close_h5file()
