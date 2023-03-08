import clip
import numpy as np
import torch
import torch.nn as nn
from gym import Space, spaces
from habitat.core.simulator import Observations
from habitat_baselines.rl.ddppo.policy import resnet
from evoenc.models.encoders.resnet_encoders import ResNetEncoder
from torch import Tensor

from evoenc.common.utils import single_frame_box_shape
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        downsample_size: int = 3,
        rgb_level: int = -1,
    ) -> None:
        super().__init__()
        self.model, self.preprocessor = clip.load(model_name)
        for param in self.model.parameters():
            param.requires_grad_(trainable)
        self.normalize_visual_inputs = True
        self.normalize_mu = torch.FloatTensor(
            [0.48145466, 0.4578275, 0.40821073]
        )
        self.normalize_sigma = torch.FloatTensor(
            [0.26862954, 0.26130258, 0.27577711]
        )
        self.rgb_embedding_seq = None
        # self.ln_rgb = nn.LayerNorm(768)
        # self.ln_text = nn.LayerNorm(512)
        self.use_mean = True
        if rgb_level == -1:
            self.model.visual.transformer.register_forward_hook(self._vit_hook)
            self.model.transformer.register_forward_hook(self._t_hook)
        else:
            self.model.visual.transformer.resblocks[
                rgb_level
            ].register_forward_hook(self._vit_hook)
            self.model.transformer.resblocks[rgb_level].register_forward_hook(
                self._t_hook
            )
        self.sub_embedding_seq = None
        self.downsample_size = downsample_size
        self.rgb_downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(downsample_size), nn.Flatten(start_dim=2)
        )

    def _normalize(self, imgs: Tensor) -> Tensor:
        if self.normalize_visual_inputs:
            device = imgs.device
            if self.normalize_sigma.device != imgs.device:
                self.normalize_sigma = self.normalize_sigma.to(device)
                self.normalize_mu = self.normalize_mu.to(device)
            imgs = (imgs / 255.0 - self.normalize_mu) / self.normalize_sigma
            imgs = imgs.permute(0, 3, 1, 2)
            return imgs
        else:
            return imgs

    def _vit_hook(self, m, i, o):
        self.rgb_embedding_seq = o.float()

    def _t_hook(self, m, i, o):
        self.sub_embedding_seq = o.float()
    def encode_sub_instruction(self, observations: Observations) -> Tensor:
        if "sub_features" in observations:
            sub_embedding = observations["sub_features"]
        else:
            with torch.no_grad():
                sub_instruction = observations["sub_instruction"].int()
                shape = sub_instruction.shape
                sub_instruction = sub_instruction.reshape((-1, shape[-1]))
                idx = (sub_instruction > 0).any(dim=-1)  # N*L, index of useful position
                attention_mask = sub_instruction > 0
                sub_embedding = torch.zeros(
                    (shape[0] * shape[1], 512), dtype=torch.float
                ).to(sub_instruction.device)
                self.model.encode_text(sub_instruction[idx]).float()
                # LND -> NLD
                sub_embedding_seq = self.sub_embedding_seq.float().permute(1, 0, 2)
                if self.use_mean:
                    am = attention_mask[idx]
                    lengths = am.sum(dim=1).unsqueeze(1) # Word numbers in useful subs
                    sub_embedding_seq = (sub_embedding_seq*am.unsqueeze(2)).sum(dim=1)/lengths
                else:
                    sub_embedding_seq = sub_embedding_seq[
                        torch.arange(sub_embedding_seq.shape[0]),
                        sub_instruction[idx].argmax(dim=-1)
                    ]
                sub_embedding[idx] = sub_embedding_seq
                sub_embedding = sub_embedding.reshape(
                    (shape[0], shape[1], sub_embedding.shape[-1])
                )
                sub_mask = (sub_embedding != 0).any(dim=2)
        return sub_embedding
    def encode_text_old(self, observations: Observations) -> Tensor:
        if "sub_features" in observations:
            sub_embedding = observations["sub_features"]
        else:
            sub_instruction = observations["sub_instruction"].int()
            bs = sub_instruction.shape[0]
            T = sub_instruction.shape[1]
            ## fast
            # N = sub_instruction.shape[2]
            sub_embedding = torch.zeros((bs, T, 512), dtype=torch.float).to(
                sub_instruction.device
            )
            for i in range(bs):
                pad = torch.zeros(
                    (1,), dtype=torch.int, device=sub_instruction.device
                )
                idx = torch.argmin(
                    torch.cat((sub_instruction[i, :, 0], pad))
                )  # effective sub instructions num
                _ = self.model.encode_text(sub_instruction[i][0:idx]).float()
                # LND -> NLD
                sub_embedding_seq = self.sub_embedding_seq.float().permute(
                    1, 0, 2
                )
                # sub_embedding_seq = self.ln_text(sub_embedding_seq)
                sub_embedding[i][0:idx] = sub_embedding_seq[
                    torch.arange(sub_embedding_seq.shape[0]),
                    sub_instruction[i][0:idx].argmax(dim=-1),
                ]
        return sub_embedding
    def encode_raw(self, observations: Observations) -> Tensor:
        if "inst_features" in observations:
            raw_embedding_seq = observations["inst_features"]
        else:
            instruction = observations["instruction"].int()
            _ = self.model.encode_text(instruction).float()
            # LND -> NLD
            raw_embedding_seq = self.sub_embedding_seq.float().permute(
                1, 0, 2
            )
            raw_embedding_seq[instruction==0] = 0
        return raw_embedding_seq

    def encode_image(
        self, observations: Observations, return_seq: bool = True
    ) -> Tensor:
        if (
            "rgb_features" in observations
            and "rgb_seq_features" in observations
        ):
            rgb_embedding = observations["rgb_features"]
            rgb_embedding_seq = observations["rgb_seq_features"]
        else:
            rgb_observations = observations["rgb"]
            _ = self.model.encode_image(
                self._normalize(rgb_observations)
            ).float()
            s = self.rgb_embedding_seq.shape
            # LND -> NLD
            rgb_embedding_seq = self.rgb_embedding_seq.float().permute(1, 0, 2)
            # rgb_embedding_seq = self.ln_rgb(rgb_embedding_seq)
            rgb_embedding = rgb_embedding_seq[:, 0, :]
            if self.downsample_size==7:
                rgb_embedding_seq = rgb_embedding_seq[:, 1:, :].permute(0, 2, 1)
            else:
                # NLD -> NDL -> NDHW
                rgb_embedding_seq = (
                    rgb_embedding_seq[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(s[1], s[2], 7, 7)
                )
                rgb_embedding_seq = self.rgb_downsample(rgb_embedding_seq)
        if return_seq:
            return (
                rgb_embedding,
                rgb_embedding_seq,
            )  # returns [BATCH x OUTPUT_DIM]
        else:
            return rgb_embedding

class InstructionEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", trainable=False, use_layer=6, tune_layer=[], use_cls=False):
        super(InstructionEncoder, self).__init__()
        self.model_name = model_name
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.use_layer = use_layer
        self.tune_layer = tune_layer
        self.use_cls = use_cls
        for param in self.bert_model.parameters():
            param.requires_grad_(trainable)
        # unfreeze final layer
        for tune in tune_layer:
            for param in self.bert_model.transformer.layer[tune].parameters():
                param.requires_grad_(True)
        if use_layer!=6:
            self.bert_model.transformer.layer[use_layer].register_forward_hook(self._t_hook)
        self.feature = None
    def _t_hook(self, m, i, o):
        self.feature = o
    def encode_instruction(self, observations):
        if "inst_features" in observations:
            inst_embedding = observations["inst_features"]
        else:
            # with torch.no_grad():
            instruction = observations["instruction"].long()
            attention_mask = instruction>0
            if "distilbert" in self.model_name:
                inputs = {"input_ids":instruction, "attention_mask":attention_mask}
            else:
                token_type_ids = torch.zeros_like(instruction, dtype=torch.int)
                inputs = {"input_ids":instruction, "token_type_ids":token_type_ids,"attention_mask":attention_mask}
            res = self.bert_model(**inputs)
            if self.use_layer==6:
                inst_embedding = res["last_hidden_state"]
            else:
                inst_embedding = self.feature[0]
            attention_mask = attention_mask.bool().logical_not()
            inst_embedding[attention_mask] = 0
            if not self.use_cls:
                inst_embedding = inst_embedding[:,1:,:]
        return inst_embedding
    def encode_sub_instruction(self, observations):
        if "sub_features" in observations:
            sub_embedding = observations["sub_features"]
            sub_mask = (sub_embedding!=0).any(dim=2)
        else:
            with torch.no_grad():
                sub_instruction = observations["sub_instruction"].long()
                shape = sub_instruction.shape
                sub_instruction = sub_instruction.reshape((-1, shape[-1]))
                idx = (sub_instruction>0).any(dim=-1)

                token_type_ids = torch.zeros_like(sub_instruction, dtype=torch.int)
                attention_mask = sub_instruction>0
                if "distilbert" in self.model_name:
                    inputs = {"input_ids":sub_instruction[idx], "attention_mask":attention_mask[idx]}
                else:
                    inputs = {"input_ids":sub_instruction[idx], "token_type_ids":token_type_ids[idx],"attention_mask":attention_mask[idx]}
                res = self.bert_model(**inputs)
                am = attention_mask[idx]
                lengths = am.sum(dim=1).unsqueeze(1)
                hidden_state = (res["last_hidden_state"]*am.unsqueeze(2)).sum(dim=1)/lengths
                sub_embedding = torch.zeros((shape[0]*shape[1], hidden_state.shape[-1]), device=sub_instruction.device)
                sub_embedding[idx] = hidden_state
                sub_embedding = sub_embedding.reshape((shape[0], shape[1], sub_embedding.shape[-1]))
                sub_mask = (sub_embedding!=0).any(dim=2)
        return sub_embedding, sub_mask

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Space,
        output_size: int = 128,
        checkpoint: str = "NONE",
        backbone: str = "resnet50",
        resnet_baseplanes: int = 32,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
        final_relu: bool = False,
    ) -> None:
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            spaces.Dict(
                {
                    "depth": single_frame_box_shape(
                        observation_space.spaces["depth"]
                    )
                }
            ),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            final_relu=final_relu,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)
        self.depth_seq_features = None
        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)
    def get_depth_seq_features(self):
        return self.depth_seq_features

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        return self.encode_depth(observations)
    def encode_depth(self, observations, embeddings=None):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_seq_features" in observations:
            x = observations["depth_seq_features"]
        else:
            x = self.visual_encoder(observations)
            self.depth_seq_features = x
        if embeddings:
            b, c, h, w = x.size()
            spatial_features = (
                embeddings(
                    torch.arange(
                        0,
                        embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, embeddings.embedding_dim, h, w)
            )
            x = torch.cat([x, spatial_features], dim=1)
        depth_seq_embedding = x.flatten(start_dim=2)
        depth_embedding = x.flatten(start_dim=1)
        return depth_embedding, depth_seq_embedding
if __name__ == "__main__":
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
    from habitat_baselines.common.tensorboard_utils import TensorboardWriter
    from habitat_baselines.utils.common import batch_obs
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_batch,
        apply_obs_transforms_obs_space,
        get_active_obs_transforms,
    )

    from evoenc.common.aux_losses import AuxLosses
    from evoenc.common.base_il_trainer import BaseVLNCETrainer
    from evoenc.common.env_utils import construct_envs
    from evoenc.common.utils import extract_instruction_tokens
    from evoenc.config.default import get_config

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=FutureWarning)
    #     import tensorflow as tf  # noqa: F401
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        required=False,
        help="path to config yaml containing info about experiment",
    )

    args = parser.parse_args()
    config = get_config(args.exp_config)
    envs = construct_envs(config, get_env_class(config.ENV_NAME))
    observations = envs.reset()

    ## test part!!
    # origin_observations = copy.deepcopy(observations)
    observations = extract_instruction_tokens(
        observations,
        config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
    )
    batch = batch_obs(observations)
    obs_transforms = get_active_obs_transforms(config)
    batch = apply_obs_transforms_batch(batch, obs_transforms)
    # instruction_encoder = CLIPEncoder(model_name="ViT-B/32",rgb_level=-2).to(batch["instruction"].device)
    instruction_encoder = InstructionEncoder()
    inst = instruction_encoder.encode_instruction(batch)