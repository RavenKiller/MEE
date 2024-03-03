import gc
import os
import sys
import random
from typing import Any, Optional, Tuple
import warnings
from collections import defaultdict
from datetime import datetime
from gym import Space
import h5py
from PIL import Image
import pickle
import numpy as np
import torch
from torcheval.metrics.functional import binary_f1_score
import json
import tqdm
import ast
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from habitat import Config, logger
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
from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    BertTokenizerFast,
    RobertaTokenizer,
    AutoTokenizer,
)

from evoenc.common.aux_losses import AuxLosses
from evoenc.common.base_il_trainer import BaseVLNCETrainer
from evoenc.common.env_utils import construct_envs
from evoenc.common.utils import extract_instruction_tokens

from accelerate import Accelerator

accelerator = Accelerator()

RAND_MIN = 25
RAND_MAX = 100000
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SCALE = 1000.0
LEN = 256
PAD_IDX = 1
SUB_PAD_IDX = 1
SUB_LEN = 50
SUB_NUM = 12


def get_warmup_scheduler(optimizer, warmup_steps):
    factor = lambda steps: steps / warmup_steps if steps < warmup_steps else 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def randint_exclude(a, b, excludes=[]):
    res = random.randint(a, b)
    return res if res not in excludes else randint_exclude(a, b, excludes)


def stage_collate_fn(batch):
    # Process different length of sub
    batch_ret = {}
    key_list = batch[0].keys()
    for k in key_list:
        datas = []
        for ele in batch:
            datas.append(ele[k])
        if k == "text":  # (B, L)
            L = max([len(data) for data in datas])
            datas = [
                F.pad(data, (0, max(0, L - len(data))), "constant", PAD_IDX)
                for data in datas
            ]
            batch_ret[k] = torch.stack(datas)
            batch_ret[k + "_mask"] = batch_ret[k] != PAD_IDX
        elif k == "sub":  # (B, SUB_NUM, SUB_LEN)
            N_NUM = max([data.shape[0] for data in datas])
            N_LEN = max([max([len(v) for v in data]) for data in datas])
            datas = [
                F.pad(
                    data,
                    (0, N_LEN - data.shape[1], 0, N_NUM - data.shape[0]),
                    "constant",
                    SUB_PAD_IDX,
                )
                for data in datas
            ]
            batch_ret[k] = torch.stack(datas)
            batch_ret[k + "_mask"] = batch_ret[k] != SUB_PAD_IDX
        elif isinstance(datas[0], torch.Tensor):
            batch_ret[k] = torch.stack(datas)
        else:
            batch_ret[k] = datas
    del batch
    return batch_ret


class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split="train",
        data_frac=0.75,
    ):
        super().__init__()
        self.config = config
        folder = Path(config.PRETRAIN.STAGE1.folder)
        if os.path.exists(folder / "stage1_files.json"):
            with open(folder / "stage1_files.json", "r") as f:
                files = json.load(f)
            self.rgb = files["rgb"]
            self.depth = files["depth"]
            self.text = files["text"]
            self.sub = files["sub"]
        else:
            self.rgb = list(folder.glob(f"rgb-*/{split}/*.jpg"))
            self.depth = list(folder.glob(f"depth-*/{split}/*.png"))
            self.text = list(folder.glob(f"text-*/{split}/*.txt"))
            self.sub = list(folder.glob(f"sub-*/{split}/*.txt"))
            files = {
                "rgb": [str(v) for v in self.rgb],
                "depth": [str(v) for v in self.depth],
                "text": [str(v) for v in self.text],
                "sub": [str(v) for v in self.sub],
            }
            with open(folder / "stage1_files.json", "w") as f:
                json.dump(files, f)
            del files
        # Solve the OOM problem
        self.rgb = np.array([str(v) for v in self.rgb]).astype(np.string_)
        self.depth = np.array([str(v) for v in self.depth]).astype(np.string_)
        self.text = np.array([str(v) for v in self.text]).astype(np.string_)
        self.sub = np.array([str(v) for v in self.sub]).astype(np.string_)

        self.data_frac = data_frac

        self.rgb_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.CLIP.model_name
        )
        self.depth_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.TAC.model_name
        )
        self.text_processor = RobertaTokenizer.from_pretrained(
            config.MODEL.BERT.model_name
        )
        self.sub_processor = AutoTokenizer.from_pretrained(
            config.MODEL.SBERT.model_name
        )

    def __len__(self):
        return int(
            self.data_frac
            * max(len(self.rgb), len(self.depth), len(self.text), len(self.sub))
        )

    def __getitem__(self, idx):
        if idx == -1:  # skip signal from sampler
            return {}
        rgb_path = self.rgb[idx % len(self.rgb)]
        depth_path = self.depth[idx % len(self.depth)]
        text_path = self.text[idx % len(self.text)]
        sub_path = self.sub[idx % len(self.sub)]

        rgb = Image.open(rgb_path)
        rgb = self.rgb_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(
            0
        )

        depth = Image.open(depth_path)
        depth = np.array(depth).astype("float32") / DEPTH_SCALE  # to meters
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)  # clip to [MIN_DEPTH, MAX_DEPTH]
        depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)  # normalize to [0,1]
        depth = np.expand_dims(depth, axis=2).repeat(3, axis=2)  # extend to 3 channels
        depth = self.depth_processor(
            depth,
            do_resize=False,
            do_center_crop=False,
            do_rescale=False,
            do_convert_rgb=False,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        with open(text_path, "r") as f:
            text = f.read()
        text = self.text_processor(
            text, return_tensors="pt", padding=True, truncation=True, max_length=LEN
        ).input_ids.squeeze(0)

        with open(sub_path, "r") as f:
            sub = f.read().split("\n")
        sub = sub[: min(len(sub), SUB_NUM)]
        sub = self.sub_processor(
            sub, return_tensors="pt", padding=True, truncation=True, max_length=SUB_LEN
        ).input_ids

        return {
            "rgb": rgb,
            "depth": depth,
            "text": text,
            "sub": sub,
        }


class Stage2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split="train",
        positive_ratio=0.33,
        data_frac=0.75,
    ):
        super().__init__()
        self.config = config
        folder = Path(config.PRETRAIN.STAGE2.folder)
        if os.path.exists(folder / "stage2_files.json"):
            with open(folder / "stage2_files.json", "r") as f:
                files = json.load(f)
            self.rgb = files["rgb"][0:1000]
            self.depth = files["depth"][0:1000]
            self.text = files["text"][0:1000]
            self.sub = files["sub"][0:1000]
        else:
            self.rgb = list(folder.glob(f"rgb-*/{split}/*.jpg"))
            self.depth = list(folder.glob(f"depth-*/{split}/*.png"))
            self.text = list(folder.glob(f"text-*/{split}/*.txt"))
            self.sub = list(folder.glob(f"sub-*/{split}/*.txt"))
            files = {
                "rgb": [str(v) for v in self.rgb],
                "depth": [str(v) for v in self.depth],
                "text": [str(v) for v in self.text],
                "sub": [str(v) for v in self.sub],
            }
            with open(folder / "stage2_files.json", "w") as f:
                json.dump(files, f)
        # Solve the OOM problem
        self.rgb = np.array([str(v) for v in self.rgb]).astype(np.string_)
        self.depth = np.array([str(v) for v in self.depth]).astype(np.string_)
        self.text = np.array([str(v) for v in self.text]).astype(np.string_)
        self.sub = np.array([str(v) for v in self.sub]).astype(np.string_)

        self.data_frac = data_frac
        self.positive_ratio = positive_ratio
        self.rgb_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.CLIP.model_name
        )
        self.depth_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.TAC.model_name
        )
        self.text_processor = RobertaTokenizer.from_pretrained(
            config.MODEL.BERT.model_name
        )
        self.sub_processor = AutoTokenizer.from_pretrained(
            config.MODEL.SBERT.model_name
        )
        assert len(self.rgb) == len(self.depth)
        assert len(self.text) == len(self.sub)

    def __len__(self):
        return int(
            self.data_frac
            * max(len(self.rgb), len(self.depth), len(self.text), len(self.sub))
        )

    def __getitem__(self, idx):
        positive = random.random() <= self.positive_ratio
        negative_idx = idx
        if not positive:
            # negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
            negative_idx = randint_exclude(0, len(self) - 1, [idx])
        rgb_path = self.rgb[idx % len(self.rgb)]
        depth_path = self.depth[negative_idx % len(self.depth)]
        text_path = self.text[idx % len(self.text)]
        sub_path = self.sub[negative_idx % len(self.sub)]

        rgb = Image.open(rgb_path)
        rgb = self.rgb_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(
            0
        )

        depth = Image.open(depth_path)
        depth = np.array(depth).astype("float32") / DEPTH_SCALE  # to meters
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)  # clip to [MIN_DEPTH, MAX_DEPTH]
        depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)  # normalize to [0,1]
        depth = np.expand_dims(depth, axis=2).repeat(3, axis=2)  # extend to 3 channels
        depth = self.depth_processor(
            depth,
            do_resize=False,
            do_center_crop=False,
            do_rescale=False,
            do_convert_rgb=False,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        with open(text_path, "r") as f:
            text = f.read()
        text = self.text_processor(
            text, return_tensors="pt", padding=True, truncation=True, max_length=LEN
        ).input_ids.squeeze(0)

        with open(sub_path, "r") as f:
            sub = f.read().split("\n")
        sub = sub[: min(len(sub), SUB_NUM)]
        sub = self.sub_processor(
            sub, return_tensors="pt", padding=True, truncation=True, max_length=SUB_LEN
        ).input_ids

        return {
            "rgb": rgb,
            "depth": depth,
            "text": text,
            "sub": sub,
            "inner_gt": torch.tensor(positive, dtype=int),
        }


class Stage3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split="train",
        positive_ratio=0.33,
        inner_positive_ratio=0.5,
        data_frac=0.75,
    ):
        super().__init__()
        self.config = config
        folder = Path(config.PRETRAIN.STAGE3.folder)
        if os.path.exists(folder / "stage3_files.json"):
            with open(folder / "stage3_files.json", "r") as f:
                files = json.load(f)
            self.rgb = files["rgb"]
            self.depth = files["depth"]
            self.text = files["text"]
            self.sub = files["sub"]
        else:
            self.rgb = sorted([str(v) for v in folder.glob(f"rgb-*/{split}/*.jpg")])
            self.depth = sorted([str(v) for v in folder.glob(f"depth-*/{split}/*.png")])
            self.text = sorted([str(v) for v in folder.glob(f"text-*/{split}/*.txt")])
            self.sub = sorted([str(v) for v in folder.glob(f"sub-*/{split}/*.txt")])
            files = {
                "rgb": self.rgb,
                "depth": self.depth,
                "text": self.text,
                "sub": self.sub,
            }
            with open(folder / "stage3_files.json", "w") as f:
                json.dump(files, f)
        # Solve the OOM problem
        self.rgb = np.array(self.rgb).astype(np.string_)
        self.depth = np.array(self.depth).astype(np.string_)
        self.text = np.array(self.text).astype(np.string_)
        self.sub = np.array(self.sub).astype(np.string_)
        self.data_frac = data_frac
        self.positive_ratio = positive_ratio
        self.inner_positive_ratio = inner_positive_ratio
        self.rgb_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.CLIP.model_name
        )
        self.depth_processor = CLIPImageProcessor.from_pretrained(
            config.MODEL.TAC.model_name
        )
        self.text_processor = RobertaTokenizer.from_pretrained(
            config.MODEL.BERT.model_name
        )
        self.sub_processor = AutoTokenizer.from_pretrained(
            config.MODEL.SBERT.model_name
        )
        assert len(self.rgb) == len(self.depth)
        assert len(self.rgb) == len(self.text)
        assert len(self.rgb) == len(self.sub)

    def __len__(self):
        return int(
            self.data_frac
            * max(len(self.rgb), len(self.depth), len(self.text), len(self.sub))
        )

    def __getitem__(self, idx):
        positive = random.random() <= self.positive_ratio
        inner_positive = random.random() <= self.inner_positive_ratio
        if positive:
            rgb_path = self.rgb[idx % len(self.rgb)]
            depth_path = self.depth[idx % len(self.depth)]
            text_path = self.text[idx % len(self.text)]
            sub_path = self.sub[idx % len(self.sub)]
        else:
            if inner_positive:
                negative_idx = randint_exclude(0, len(self) - 1, [idx])
                rgb_path = self.rgb[idx % len(self.rgb)]
                depth_path = self.depth[idx % len(self.depth)]
                text_path = self.text[negative_idx % len(self.text)]
                sub_path = self.sub[negative_idx % len(self.sub)]
            else:
                rgb_path = self.rgb[idx % len(self.rgb)]
                negative_idx1 = randint_exclude(0, len(self) - 1, [idx])
                depth_path = self.depth[negative_idx1 % len(self.depth)]

                negative_idx2 = randint_exclude(0, len(self) - 1, [])
                text_path = self.text[negative_idx2 % len(self.text)]
                negative_idx3 = randint_exclude(0, len(self) - 1, [negative_idx2])
                sub_path = self.sub[negative_idx3 % len(self.sub)]
        print(positive, inner_positive)
        if positive:
            print(idx)
        elif inner_positive:
            print(idx, negative_idx)
        else:
            print(idx, negative_idx1, negative_idx2, negative_idx3)

        rgb = Image.open(rgb_path)
        rgb = self.rgb_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(
            0
        )

        depth = Image.open(depth_path)
        depth = np.array(depth).astype("float32") / DEPTH_SCALE  # to meters
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)  # clip to [MIN_DEPTH, MAX_DEPTH]
        depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)  # normalize to [0,1]
        depth = np.expand_dims(depth, axis=2).repeat(3, axis=2)  # extend to 3 channels
        depth = self.depth_processor(
            depth,
            do_resize=False,
            do_center_crop=False,
            do_rescale=False,
            do_convert_rgb=False,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        with open(text_path, "r") as f:
            text = f.read()
        text = self.text_processor(
            text, return_tensors="pt", padding=True, truncation=True, max_length=LEN
        ).input_ids.squeeze(0)

        with open(sub_path, "r") as f:
            sub = f.read().split("\n")
        sub = sub[: min(len(sub), SUB_NUM)]
        sub = self.sub_processor(
            sub, return_tensors="pt", padding=True, truncation=True, max_length=SUB_LEN
        ).input_ids

        return {
            "rgb": rgb,
            "depth": depth,
            "text": text,
            "sub": sub,
            "inner_gt": torch.tensor(
                np.logical_or(positive, inner_positive), dtype=int
            ),
            "outer_gt": torch.tensor(positive, dtype=int),
        }


class SkipRandomSampler(torch.utils.data.RandomSampler):
    def __init__(
        self, data_source, replacement=False, num_samples=None, skip_samples=0
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.skip_samples = skip_samples

        if self._num_samples is not None and replacement is False:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integeral "
                "value, but got num_samples={}".format(self.num_samples)
            )
        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )
        res = torch.randperm(n)
        res[: self.skip_samples] = -1
        return iter(res.tolist())


@baseline_registry.register_trainer(name="evopretrainer")
class PreTrainer(BaseVLNCETrainer):
    def _make_dirs(self) -> None:
        r"""Makes directories for log files, checkpoints & results."""
        self._make_ckpt_dir()
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _get_spaces(self, config, envs=None) -> Tuple[Space]:
        if os.path.exists("habitat_extensions/observation_space.pkl"):
            with open("habitat_extensions/observation_space.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open("habitat_extensions/action_space.pkl", "rb") as f:
                action_space = pickle.load(f)
            return observation_space, action_space
        else:
            observation_space, action_space = super()._get_spaces(config, envs)
            with open("habitat_extensions/observation_space.pkl", "wb") as f:
                pickle.dump(observation_space, f)
            with open("habitat_extensions/action_space.pkl", "wb") as f:
                pickle.dump(action_space, f)
            return observation_space, action_space

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
        assert config.PRETRAIN.stage in ["STAGE1", "STAGE2", "STAGE3"]
        if config.PRETRAIN.stage == "STAGE1":
            stage_config = config.PRETRAIN.STAGE1
        elif config.PRETRAIN.stage == "STAGE2":
            stage_config = config.PRETRAIN.STAGE2
        elif config.PRETRAIN.stage == "STAGE3":
            stage_config = config.PRETRAIN.STAGE3
        self.stage_config = stage_config
        lr = stage_config.lr
        load_from_ckpt = stage_config.load_from_ckpt
        ckpt_to_load = stage_config.ckpt_to_load
        self.stage_config = stage_config

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = get_warmup_scheduler(self.optimizer, stage_config.warmup)
        if load_from_ckpt and train_mode:
            ckpt_path = ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict_woenc(
                ckpt_dict["state_dict"], excludes=config.PRETRAIN.excludes
            )
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
            if config.PRETRAIN.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optimizer_state"])
                self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
                self.step_id = ckpt_dict["step_id"]
                logger.info(f"Resume training from checkpoint: {ckpt_path}")
                logger.info(f"Start training from step: {self.step_id+1}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")
        self.max_grad_norm = config.PRETRAIN.max_grad_norm

    def save_checkpoint(self, file_name: str, step_id: int = 0) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.policy.state_dict_woenc(
                excludes=self.config.PRETRAIN.excludes
            ),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step_id": step_id,
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
        if self.config.PRETRAIN.stage == "STAGE1":
            self._train_stage1()
        elif self.config.PRETRAIN.stage == "STAGE2":
            self._train_stage2()
        elif self.config.PRETRAIN.stage == "STAGE3":
            self._train_stage3()

    def _train_stage1(self):
        dataset = Stage1Dataset(
            config=self.config,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=stage_collate_fn,
            pin_memory=False,
            sampler=SkipRandomSampler(
                dataset, skip_samples=(self.step_id + 1) * self.stage_config.batch_size
            ),
        )

        self.policy, self.optimizer, dataloader, self.scheduler = accelerator.prepare(
            self.policy, self.optimizer, dataloader, self.scheduler
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
                total=len(dataloader.dataset) // self.stage_config.batch_size,
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                if iter_num <= self.step_id:  # skip steps in the ckpt
                    iter_num += 1
                    continue
                self.optimizer.zero_grad()
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                losses = self.policy.net.stage1_forward(batch)
                total_loss = 0
                for i, k in enumerate(losses):
                    w = self.stage_config.loss_weights[i]
                    total_loss += w * losses[k]
                # total_loss.backward()
                accelerator.backward(total_loss)
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()

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
                if iter_num % self.stage_config.save_steps == 0:
                    self.save_checkpoint(
                        f"ckpt.{self.config.MODEL.policy_name}.step{iter_num}.pth",  # to continue train
                        step_id=iter_num,
                    )
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.epoch{epoch}.pth",  # to continue train
                step_id=iter_num,
            )
        writer.close()

    def _train_stage2(self):
        dataset = Stage2Dataset(
            config=self.config,
            positive_ratio=self.stage_config.positive_ratio,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=stage_collate_fn,
            # pin_memory=True
        )
        self.policy, self.optimizer, dataloader, self.scheduler = accelerator.prepare(
            self.policy, self.optimizer, dataloader, self.scheduler
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
                total=len(dataloader.dataset) // self.stage_config.batch_size,
                leave=False,
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                if iter_num <= self.step_id:
                    iter_num += 1
                    continue
                self.optimizer.zero_grad()
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                losses, _ = self.policy.net.stage2_forward(batch)
                total_loss = 0
                for i, k in enumerate(losses):
                    w = self.stage_config.loss_weights[i]
                    total_loss += w * (losses[k])
                # total_loss.backward()
                accelerator.backward(total_loss)
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()

                batch_bar.set_description(f"E {epoch}.")
                batch_bar.set_postfix(
                    {
                        "loss": "%2.4f" % (total_loss),
                        "rec": "%2.3f" % (losses["loss_rec"]),
                        "mea": "%2.3f" % (losses["loss_mean"]),
                        "ali": "%2.3f" % (losses["loss_align"]),
                    }
                )
                for k in losses:
                    writer.add_scalar("loss/%s" % (k), losses[k], iter_num)
                writer.add_scalar("loss/total", total_loss, iter_num)
                if iter_num % self.stage_config.save_steps == 0:
                    self.save_checkpoint(
                        f"ckpt.{self.config.MODEL.policy_name}.step{iter_num}.pth",  # to continue train
                        step_id=iter_num,
                    )
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.epoch{epoch}.pth"  # to continue train
            )
        writer.close()

    def _train_stage3(self):
        dataset = Stage3Dataset(
            folder=self.stage_config.folder,
            positive_ratio=self.stage_config.positive_ratio,
            inner_positive_ratio=self.stage_config.inner_positive_ratio,
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
                + "stage3",
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
                losses, _ = self.policy.net.stage3_forward(batch)
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

        if self.config.PRETRAIN.stage == "STAGE1":
            pass
        elif self.config.PRETRAIN.stage == "STAGE2":
            self._eval_stage2(checkpoint_index)
        elif self.config.PRETRAIN.stage == "STAGE3":
            self._eval_stage3(checkpoint_index)

    def _eval_stage2(self, checkpoint_index):
        dataset = Stage2Dataset(
            folder=self.stage_config.folder,
            positive_ratio=self.stage_config.positive_ratio,
            data_frac=1.0,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
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
            f"stats_ckpt_{checkpoint_index}_stage2.json",
        )
        if os.path.exists(fname):
            logger.info("skipping -- evaluation exists.")
            return
        pred_v = []
        gt_v = []
        pred_l = []
        gt_l = []
        with torch.no_grad():
            for batch in batch_bar:
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                _, res = self.policy.net.stage2_forward(batch)
                gt_v.append(res["align_gt_v"].squeeze().cpu())
                pred_v.append(torch.sigmoid(res["align_pre_v"].squeeze().cpu()))
                gt_l.append(res["align_gt_l"].squeeze().cpu())
                pred_l.append(torch.sigmoid(res["align_pre_l"].squeeze().cpu()))
                batch_bar.set_description(f"C {checkpoint_index}.")
                batch_bar.set_postfix(
                    {
                        "V": "%2.4f" % (binary_f1_score(pred_v[-1], gt_v[-1])),
                        "L": "%2.4f" % (binary_f1_score(pred_l[-1], gt_l[-1])),
                    }
                )
        pred_v = torch.cat(pred_v)
        gt_v = torch.cat(gt_v)
        f1_score_v = binary_f1_score(pred_v, gt_v)
        pred_l = torch.cat(pred_l)
        gt_l = torch.cat(gt_l)
        f1_score_l = binary_f1_score(pred_l, gt_l)
        f1_score = (f1_score_l + f1_score_v) / 2
        print(f1_score_v, f1_score_l, f1_score)
        with open(fname, "w") as f:
            json.dump({"f1_socre": float(f1_score)}, f, indent=4)

    def _eval_stage3(self, checkpoint_index):
        dataset = Stage3Dataset(
            folder=self.stage_config.folder,
            positive_ratio=self.stage_config.positive_ratio,
            inner_positive_ratio=self.stage_config.inner_positive_ratio,
            data_frac=1.0,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stage_config.batch_size,
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
            f"stats_ckpt_{checkpoint_index}_stage3.json",
        )
        if os.path.exists(fname):
            logger.info("skipping -- evaluation exists.")
            return
        pred = []
        gt = []
        with torch.no_grad():
            for batch in batch_bar:
                batch = {k: v.to(device=self.device) for k, v in batch.items()}
                _, res = self.policy.net.stage3_forward(batch)
                gt.append(res["outer_gt"].squeeze())
                pred.append(torch.sigmoid(res["outer_pre"].squeeze()))
                batch_bar.set_description(f"C {checkpoint_index}.")
                batch_bar.set_postfix(
                    {
                        "F1": "%2.4f" % (binary_f1_score(pred[-1], gt[-1])),
                    }
                )
        pred = torch.cat(pred).cpu()
        gt = torch.cat(gt).cpu()
        f1_score = binary_f1_score(pred, gt)
        with open(fname, "w") as f:
            json.dump({"f1_socre": float(f1_score)}, f, indent=4)


if __name__ == "__main__":
    d = Stage2Dataset("/hy-tmp/stage2/rgb_depth.mat")
    loader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=8)
    for v in loader:
        print(v.keys())
        print(v["depth"].shape)
        print(type(v))
        break
