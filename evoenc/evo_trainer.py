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
from transformers import AutoProcessor, CLIPImageProcessor, BertTokenizerFast

from evoenc.common.aux_losses import AuxLosses
from evoenc.common.base_il_trainer import BaseVLNCETrainer
from evoenc.common.env_utils import construct_envs
from evoenc.common.utils import extract_instruction_tokens

RAND_MIN = 25
RAND_MAX = 100000
MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SCALE = 1000.0
PAD_IDX = 0
SUB_LEN = 128
SUB_NUM = 32

def stage1_collate_fn(batch):
    # Process different length of videos
    batch_ret = {}
    key_list = batch[0].keys()
    for k in key_list:
        datas = []
        for ele in batch:
            datas.append(ele[k])
        B = len(datas)
        if k=="text": # (B, L)
            L = max([len(data) for data in datas])
            datas = [F.pad(data, (0, max(0, L-len(data))), "constant", PAD_IDX) for data in datas]
            batch_ret[k] = torch.stack(datas)
            batch_ret[k+"_mask"] = (batch_ret[k]>0)
        elif k=="sub": # (B, SUB_NUM, SUB_LEN)
            N_NUM = max([data.shape[0] for data in datas])
            N_LEN = max([max([len(v) for v in data]) for data in datas])
            datas = [F.pad(data, (0, N_LEN-data.shape[1], 0, N_NUM-data.shape[0]), "constant", PAD_IDX) for data in datas]
            batch_ret[k] = torch.stack(datas)
            batch_ret[k+"_mask"] = (batch_ret[k]>0)
        elif isinstance(datas[0], torch.Tensor):
            batch_ret[k] = torch.stack(datas)
        else:
            batch_ret[k] = datas
    return batch_ret
class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split="train",
    ):
        super().__init__()
        self.config = config
        folder = Path(config.PRETRAIN.STAGE1.folder)
        if os.path.exists(folder/"stage1_files.json"):
            with open(folder/"stage1_files.json", "r") as f:
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
            with open(folder/"stage1_files.json", "w") as f:
                json.dump(files, f)

        self.rgb_processor = CLIPImageProcessor.from_pretrained(config.MODEL.CLIP.model_name)
        self.depth_processor = CLIPImageProcessor.from_pretrained(
                config.MODEL.TAC.model_name
            )
        self.text_processor = BertTokenizerFast.from_pretrained(config.MODEL.BERT.model_name)
        self.sub_processor = self.text_processor

    def __len__(self):
        return max(len(self.rgb), len(self.depth), len(self.text), len(self.sub))

    def __getitem__(self, idx):
        rgb_path = self.rgb[idx % len(self.rgb)]
        depth_path = self.depth[idx % len(self.depth)]
        text_path = self.text[idx % len(self.text)]
        sub_path = self.sub[idx % len(self.sub)]

        rgb = Image.open(rgb_path)
        rgb = self.rgb_processor(images=rgb, return_tensors="pt").pixel_values.squeeze(0)

        depth = Image.open(depth_path)
        depth = (
            np.array(depth).astype("float32") / DEPTH_SCALE
        )  # to meters
        depth = np.clip(
            depth, MIN_DEPTH, MAX_DEPTH
        )  # clip to [MIN_DEPTH, MAX_DEPTH]
        depth = (depth - MIN_DEPTH) / (
            MAX_DEPTH - MIN_DEPTH
        )  # normalize to [0,1]
        depth = np.expand_dims(depth, axis=2).repeat(
            3, axis=2
        )  # extend to 3 channels
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
        text = self.text_processor(text, return_tensors="pt", truncation=True).input_ids.squeeze(0)

        with open(sub_path, "r") as f:
            sub = f.read().split("\n")
        sub = sub[:min(len(sub),SUB_NUM)]
        sub = self.sub_processor(sub, return_tensors="pt", padding=True, truncation=True, max_length=SUB_LEN).input_ids

        return {
            "rgb": rgb,
            "depth": depth,
            "text": text,
            "sub": sub,
        }



class Stage2Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, positive_ratio=0.33, data_frac=1.0):
        super().__init__()
        self.vision_handler = h5py.File(
            os.path.join(folder, "rgb_depth_large.mat"), "r"
        )
        self.rgb = self.vision_handler["rgb"]
        self.depth = self.vision_handler["depth"]
        self.vision_num = self.rgb.shape[0]

        self.language_handler = h5py.File(
            os.path.join(folder, "inst_sub_large.mat"), "r"
        )
        self.instructions = self.language_handler["instructions"]
        self.sub_instructions = self.language_handler["sub_instructions"]
        self.language_num = self.instructions.shape[0]

        self.positive_ratio = positive_ratio
        self.data_frac = data_frac

    def __len__(self):
        return int(max(self.vision_num, self.language_num) * self.data_frac)

    def __getitem__(self, idx):
        positive = random.random() <= self.positive_ratio
        negative_idx = idx
        if not positive:
            negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
        rgb = self.rgb[idx % self.vision_num]
        depth = self.depth[negative_idx % self.vision_num]
        instruction = self.instructions[idx % self.language_num]
        sub_instruction = self.sub_instructions[negative_idx % self.language_num]

        return {
            "rgb": rgb.astype(np.float32),
            "depth": depth.astype(np.float32),
            "instruction": instruction.astype(np.int32),  # do not support uint32
            "sub_instruction": sub_instruction.astype(np.int32),
            "inner_gt": np.array(positive, dtype=np.int32),
        }

    def close_h5file(self):
        self.vision_handler.close()
        self.language_handler.close()


class Stage3Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, positive_ratio=0.33, inner_ratio=0.5, data_frac=1.0):
        super().__init__()
        # self.data_handler = h5py.File(os.path.join(folder, "data.mat"), "r")
        self.rgb_handler = h5py.File(os.path.join(folder, "data_rgb.mat"), "r")
        self.depth_handler = h5py.File(os.path.join(folder, "data_depth.mat"), "r")
        self.inst_handler = h5py.File(os.path.join(folder, "data_inst.mat"), "r")
        self.sub_handler = h5py.File(os.path.join(folder, "data_sub.mat"), "r")
        self.rgb = self.rgb_handler["rgb"]
        self.depth = self.depth_handler["depth"]
        self.instructions = self.inst_handler["instructions"]
        self.sub_instructions = self.sub_handler["sub_instructions"]

        self.rgb_num = self.rgb.shape[0]
        self.depth_num = self.depth.shape[0]
        self.inst_num = self.instructions.shape[0]
        self.sub_num = self.instructions.shape[0]
        assert self.rgb_num==self.depth_num
        assert self.rgb_num==self.inst_num
        assert self.inst_num==self.sub_num

        self.positive_ratio = positive_ratio  # the positive ratio
        self.inner_ratio = inner_ratio  # the negative inner alignment ratio
        self.data_frac = data_frac

    def __len__(self):
        return int(self.rgb_num * self.data_frac)

    def __getitem__(self, idx):
        positive = random.random() <= self.positive_ratio
        inner_negative = random.random() <= self.inner_ratio
        if positive:
            rgb = self.rgb[idx % self.rgb_num]
            depth = self.depth[idx % self.depth_num]
            instruction = self.instructions[idx % self.inst_num]
            sub_instruction = self.sub_instructions[idx % self.sub_num]
        else:
            if inner_negative:
                rgb = self.rgb[idx % self.rgb_num]
                negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
                depth = self.depth[negative_idx % self.depth_num]
                negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
                instruction = self.instructions[negative_idx % self.inst_num]
                negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
                sub_instruction = self.sub_instructions[negative_idx % self.sub_num]
            else:
                negative_idx = idx + random.randint(RAND_MIN, RAND_MAX)
                rgb = self.rgb[idx % self.rgb_num]
                depth = self.depth[idx % self.depth_num]
                instruction = self.instructions[negative_idx % self.inst_num]
                sub_instruction = self.sub_instructions[negative_idx % self.sub_num]

        return {
            "rgb": rgb.astype(np.float32),
            "depth": depth.astype(np.float32),
            "instruction": instruction.astype(np.int32),  # do not support uint32
            "sub_instruction": sub_instruction.astype(np.int32),
            "inner_gt": np.array((positive or not inner_negative), dtype=np.int32),
            "outer_gt": np.array(positive, dtype=np.int32),
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
    def _get_spaces(self, config, envs = None) -> Tuple[Space]:
        # return super()._get_spaces(config, envs)
        with open("habitat_extensions/observation_space.pkl","rb") as f:
            observation_space = pickle.load(f)
        with open("habitat_extensions/action_space.pkl","rb") as f:
            action_space = pickle.load(f)
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
        assert config.PRETRAIN.stage in ["STAGE1","STAGE2","STAGE3"]
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
        # with open("habitat_extensions/observation_space.pkl","wb") as f:
        #     pickle.dump(observation_space, f)
        # with open("habitat_extensions/action_space.pkl","wb") as f:
        #     pickle.dump(action_space, f)
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
            shuffle=True,
            num_workers=16,
            collate_fn = stage1_collate_fn,
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
            #     losses = self.policy.net.stage1_forward(batch)
            #     total_loss = 0
            #     for i, k in enumerate(losses):
            #         w = self.stage_config.loss_weights[i]
            #         total_loss += w * losses[k]
            #     total_loss.backward()
            #     if self.max_grad_norm:
            #         torch.nn.utils.clip_grad_norm_(
            #             self.policy.parameters(), self.max_grad_norm
            #         )
            #     self.optimizer.step()

            #     batch_bar.set_description(f"E {epoch}.")
            #     batch_bar.set_postfix(
            #         {
            #             "loss": "%2.4f" % (total_loss),
            #             "rec": "%2.3f" % (losses["loss_rec"]),
            #             "mea": "%2.3f" % (losses["loss_mean"]),
            #         }
            #     )
            #     for k in losses:
            #         writer.add_scalar("loss/%s" % (k), losses[k], iter_num)
            #     writer.add_scalar("loss/total", total_loss, iter_num)
            #     iter_num += 1
            # self.save_checkpoint(
            #     f"ckpt.{self.config.MODEL.policy_name}.{epoch}.pth"  # to continue train
            # )
        writer.close()
        dataset.close_h5file()

    def _train_stage2(self):
        dataset = Stage2Dataset(
            folder=self.stage_config.folder,
            positive_ratio=self.stage_config.positive_ratio,
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
                iter_num += 1
            self.save_checkpoint(
                f"ckpt.{self.config.MODEL.policy_name}.{epoch}.pth"  # to continue train
            )
        writer.close()
        dataset.close_h5file()

    def _train_stage3(self):
        dataset = Stage3Dataset(
            folder=self.stage_config.folder,
            positive_ratio=self.stage_config.positive_ratio,
            inner_ratio=self.stage_config.inner_ratio,
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
            inner_ratio=self.stage_config.inner_ratio,
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
