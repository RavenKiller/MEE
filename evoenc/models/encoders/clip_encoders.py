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
from PIL import Image
import requests
from transformers import CLIPVisionModel, BertModel


class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = CLIPVisionModel.from_pretrained(config.MODEL.CLIP.model_name)

    def encode_image(
        self, observations: Observations, return_seq: bool = True
    ) -> Tensor:
        if "rgb_seq_features" in observations:
            rgb_seq_features = observations["rgb_seq_features"]
        else:
            rgb_observations = observations["rgb"]
            rgb_seq_features = self.model(
                pixel_values=rgb_observations
            ).last_hidden_state
        return rgb_seq_features


class TACDepthEncoder(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = CLIPVisionModel.from_pretrained(config.MODEL.TAC.model_name)

    def encode_depth(
        self, observations: Observations, return_seq: bool = True
    ) -> Tensor:
        if "depth_seq_features" in observations:
            depth_seq_features = observations["depth_seq_features"]
        else:
            depth_observations = observations["depth"]
            depth_seq_features = self.model(
                pixel_values=depth_observations
            ).last_hidden_state
        return depth_seq_features


class BERTEncoder(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = BertModel.from_pretrained(config.MODEL.BERT.model_name)

    def encode_inst(
        self, observations: Observations, return_seq: bool = True
    ) -> Tensor:
        if "inst_seq_features" in observations:
            inst_seq_features = observations["inst_seq_features"]
        else:
            inst_observations = observations.get("instruction", None)
            inst_mask = observations.get("instruction_mask", None)
            if inst_observations is None:
                inst_observations = observations.get("text", None)
                inst_mask = observations.get("text_mask", None)

            inst_seq_features = self.model(
                input_ids=inst_observations, attention_mask=inst_mask
            ).last_hidden_state
        return inst_seq_features


class InstructionEncoder(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        trainable=False,
        use_layer=6,
        tune_layer=[],
        use_cls=False,
    ):
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
        if use_layer != 6:
            self.bert_model.transformer.layer[use_layer].register_forward_hook(
                self._t_hook
            )
        self.feature = None

    def _t_hook(self, m, i, o):
        self.feature = o

    def encode_instruction(self, observations):
        if "inst_features" in observations:
            inst_embedding = observations["inst_features"]
        else:
            # with torch.no_grad():
            instruction = observations["instruction"].long()
            attention_mask = instruction > 0
            if "distilbert" in self.model_name:
                inputs = {"input_ids": instruction, "attention_mask": attention_mask}
            else:
                token_type_ids = torch.zeros_like(instruction, dtype=torch.int)
                inputs = {
                    "input_ids": instruction,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
            res = self.bert_model(**inputs)
            if self.use_layer == 6:
                inst_embedding = res["last_hidden_state"]
            else:
                inst_embedding = self.feature[0]
            attention_mask = attention_mask.bool().logical_not()
            inst_embedding[attention_mask] = 0
            if not self.use_cls:
                inst_embedding = inst_embedding[:, 1:, :]
        return inst_embedding

    def encode_sub_instruction(self, observations):
        if "sub_features" in observations:
            sub_embedding = observations["sub_features"]
            sub_mask = (sub_embedding != 0).any(dim=2)
        else:
            with torch.no_grad():
                sub_instruction = observations["sub_instruction"].long()
                shape = sub_instruction.shape
                sub_instruction = sub_instruction.reshape((-1, shape[-1]))
                idx = (sub_instruction > 0).any(dim=-1)

                token_type_ids = torch.zeros_like(sub_instruction, dtype=torch.int)
                attention_mask = sub_instruction > 0
                if "distilbert" in self.model_name:
                    inputs = {
                        "input_ids": sub_instruction[idx],
                        "attention_mask": attention_mask[idx],
                    }
                else:
                    inputs = {
                        "input_ids": sub_instruction[idx],
                        "token_type_ids": token_type_ids[idx],
                        "attention_mask": attention_mask[idx],
                    }
                res = self.bert_model(**inputs)
                am = attention_mask[idx]
                lengths = am.sum(dim=1).unsqueeze(1)
                hidden_state = (res["last_hidden_state"] * am.unsqueeze(2)).sum(
                    dim=1
                ) / lengths
                sub_embedding = torch.zeros(
                    (shape[0] * shape[1], hidden_state.shape[-1]),
                    device=sub_instruction.device,
                )
                sub_embedding[idx] = hidden_state
                sub_embedding = sub_embedding.reshape(
                    (shape[0], shape[1], sub_embedding.shape[-1])
                )
                sub_mask = (sub_embedding != 0).any(dim=2)
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
                {"depth": single_frame_box_shape(observation_space.spaces["depth"])}
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
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--mode",
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
