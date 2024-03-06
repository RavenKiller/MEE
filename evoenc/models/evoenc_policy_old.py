from typing import Dict, Tuple
from collections import OrderedDict
import math
from habitat import logger
import sys

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry

# from habitat_baselines.rl.models.rnn_state_encoder import (
#     build_rnn_state_encoder,
# )
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from evoenc.common.aux_losses import AuxLosses

# from vlnce_baselines.models.encoders import resnet_encoders
from evoenc.models.encoders.transformer_encoder import Transformer, LayerNorm
from evoenc.models.encoders.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from evoenc.models.policy import ILPolicy

CLS = 0
DEP = 1
INS = 2
SUB = 3
EPS = 1e-12
COEF_REC_INST = 1.0  # scale the reconstruction loss
COEF_REC_SUB = 1.0  # scale the reconstruction loss


def positionalencoding1d(length, d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


@baseline_registry.register_policy
class EEPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: Config,
    ) -> None:
        super().__init__(
            EENet(
                observation_space=observation_space,
                config=config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: Space, action_space: Space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    def load_state_dict_woenc(
        self,
        state_dict: "OrderedDict[str, Tensor]",
        strict: bool = True,
        excludes: list = ["clip_encoder", "depth_encoder"],
    ):
        """Load state dict without pre-trained encoders"""
        state_dict_ret = self.state_dict()
        for k in state_dict.keys():
            if (
                k.split(".")[1] not in excludes
                and k in state_dict_ret
                and state_dict_ret[k].shape == state_dict[k].shape
            ):
                state_dict_ret[k] = state_dict[k]
            else:
                logger.info(f"Not loading: {k}")
        return self.load_state_dict(state_dict_ret, strict=strict)

    def state_dict_woenc(self, excludes: list = ["clip_encoder", "depth_encoder"]):
        """Save state dict without pre-trained encoders"""
        state_dict_ret = self.state_dict()
        keys = list(state_dict_ret.keys())
        for exclude in excludes:
            for k in keys:
                if exclude in k:
                    del state_dict_ret[k]
        return state_dict_ret


class EENet(Net):
    def __init__(
        self, observation_space: Space, config: Config, num_actions: int
    ) -> None:
        super().__init__()
        self.config = config
        model_config = config.MODEL
        self.model_config = model_config
        self._output_size = model_config.EVOENC.hidden_size
        self._hidden_size = model_config.EVOENC.hidden_size
        self._is_blind = False
        self._transformer_layers = model_config.EVOENC.layers
        self._transformer_heads = model_config.EVOENC.heads
        self._inner_dropout = model_config.EVOENC.inner_dropout
        self._masked_feature_ratio = config.PRETRAIN.masked_feature_ratio

        # Init the CLIP rgb encoder
        self.clip_encoder = clip_encoders.CLIPVisionEncoder(config)
        # Init the TAC depth encoder
        self.tac_encoder = clip_encoders.TACDepthEncoder(config)
        # Init the BERT text encoder
        self.bert_encoder = clip_encoders.BERTEncoder(config)

        self.window_size = model_config.EVOENC.window_size
        self.rgb_len = model_config.EVOENC.rgb_len
        self.depth_len = model_config.EVOENC.depth_len
        self.instruction_len = model_config.EVOENC.instruction_len
        self.sub_len = model_config.EVOENC.sub_len
        self.pe_type = model_config.EVOENC.pe_type
        self.storage = None

        self.rgb_fc = nn.Sequential(
            nn.LayerNorm(model_config.CLIP.vit_size),
            nn.Dropout(p=self.model_config.EVOENC.dropout),
            nn.Linear(self.model_config.CLIP.vit_size, self._hidden_size),
            nn.ReLU(inplace=False),
        )
        self.depth_fc = nn.Sequential(
            nn.Dropout(p=self.model_config.EVOENC.dropout),
            nn.Linear(self.model_config.DEPTH_ENCODER.single_size, self._hidden_size),
            nn.ReLU(inplace=False),
        )
        self.inst_fc = nn.Sequential(
            nn.LayerNorm(model_config.CLIP.output_size),
            nn.Dropout(p=self.model_config.EVOENC.dropout),
            nn.Linear(self.model_config.CLIP.output_size, self._hidden_size),
            nn.ReLU(inplace=False),
        )
        self.sub_fc = nn.Sequential(
            nn.LayerNorm(model_config.CLIP.output_size),
            nn.Dropout(p=self.model_config.EVOENC.dropout),
            nn.Linear(self.model_config.CLIP.output_size, self._hidden_size),
            nn.ReLU(inplace=False),
        )

        # scale = self._hidden_size**-0.5
        self.total_len = (
            4 + self.depth_len + self.rgb_len + self.instruction_len + self.sub_len
        )
        if model_config.EVOENC.learnable_pe:
            self.positional_embedding = nn.Parameter(
                torch.empty(self.total_len, self._hidden_size)
            )
            nn.init.normal_(self.positional_embedding, std=0.01)
        else:
            self.positional_embedding = nn.Parameter(
                positionalencoding1d(self.total_len, self._hidden_size),
                requires_grad=False,
            )
        self.type_embedding = nn.Parameter(torch.empty(4, self._hidden_size))
        # 0:[rgb], 1:[dep], 2:[inst], 3:[sub]
        self.token_embedding = nn.Parameter(torch.empty(4, self._hidden_size))

        if self.model_config.EVOENC.pre_ln:
            self.pre_ln = nn.LayerNorm(self._hidden_size)
            self.pre_dropout = nn.Dropout(p=self.model_config.EVOENC.pre_dropout)
        self.transformer = Transformer(
            width=self._hidden_size,
            layers=self._transformer_layers,
            heads=self._transformer_heads,
            dropout=self._inner_dropout,
        )
        if self.model_config.EVOENC.post_ln:
            self.post_ln = nn.LayerNorm(self._hidden_size)
            self.post_dropout = nn.Dropout(p=self.model_config.EVOENC.post_dropout)

        if model_config.STATE_ENCODER.num_layers_action == 1:
            dropout_ratio_rnn = 0.0
        else:
            dropout_ratio_rnn = model_config.STATE_ENCODER.dropout_ratio

        # Init action embedding
        rnn_input_size = self._hidden_size
        if model_config.EVOENC.prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.action_rgb_decoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self.action_depth_decoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self.action_inst_decoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self.action_sub_decoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self._num_recurrent_layers = self.action_rgb_decoder.num_recurrent_layers * 4
        self.s1 = self.action_rgb_decoder.num_recurrent_layers
        self.s2 = self.action_rgb_decoder.num_recurrent_layers * 2
        self.s3 = self.action_rgb_decoder.num_recurrent_layers * 3
        self.s4 = self.action_rgb_decoder.num_recurrent_layers * 4
        input_size = self._hidden_size
        # self.aggregate_ln = nn.LayerNorm(input_size)
        self.aggregate_ln = nn.LayerNorm(input_size)
        if self.model_config.EVOENC.aggregate == "cat":
            self._output_size = self._hidden_size * 4
            self.aggregate_ln = nn.Identity()

        self.rgb_features = None
        self.depth_features = None
        self.inst_features = None
        self.sub_features = None

        self.attn_mask = None

        # Init the progress monitor
        self.progress_monitor = nn.Linear(self._output_size, 1)
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
            nn.init.constant_(self.progress_monitor.bias, 0)

        # For pretrain, define some heads
        if self.model_config.EVOENC.learnable_mask:
            self.mask_embedding = nn.Parameter(torch.empty(1, 768))
        if self.config.PRETRAIN.stage != "NONE":
            # masked feature reconstruction
            self.rgb_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.vit_size
            )
            self.depth_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.DEPTH_ENCODER.single_size
            )
            self.inst_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.output_size
            )
            self.sub_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.output_size
            )
            # mean feature reconstruction
            self.mean_rgb_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.vit_size
            )
            self.mean_depth_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.DEPTH_ENCODER.single_size
            )
            self.mean_inst_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.output_size
            )
            self.mean_sub_reconstruction = nn.Linear(
                self._hidden_size, self.model_config.CLIP.output_size
            )
            # feature type prediction
            # self.type_prediction = nn.Linear(self._hidden_size, 4)
            # feature alignment
            self.inner_alignment = nn.Linear(self._hidden_size * 2, 1)
            self.outer_alignment = nn.Linear(self._hidden_size * 4, 1)
            # noise detection
            # self.noise_detection = nn.Linear(self._hidden_size, 1)
        if model_config.EVOENC.freeze_weights >= 0:
            level = model_config.EVOENC.freeze_weights
            for param in [
                self.positional_embedding,
                self.type_embedding,
                self.token_embedding,
            ]:
                param.requires_grad_(False)
            for fc in [self.rgb_fc, self.depth_fc, self.inst_fc, self.sub_fc]:
                for param in fc.parameters():
                    param.requires_grad_(False)
            if level >= 1:
                for param in self.pre_ln.parameters():
                    param.requires_grad_(False)
                for i in range(level):
                    for param in self.transformer.resblocks[i].parameters():
                        param.requires_grad_(False)

        self._init_params()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self._is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self._num_recurrent_layers

    def get_rgb_features(self):
        return self.rgb_features

    def get_rgb_seq_features(self):
        return self.rgb_seq_features

    def get_depth_seq_features(self):
        return self.depth_seq_features

    def get_inst_features(self):
        return self.inst_features

    def get_sub_features(self):
        return self.sub_features

    def _init_params(self):
        nn.init.normal_(self.token_embedding, std=0.01)
        nn.init.normal_(self.type_embedding, std=0.01)
        if self.model_config.EVOENC.learnable_mask:
            nn.init.normal_(self.mask_embedding, std=0.02)
        proj_std = (self._hidden_size**-0.5) * ((2 * self._transformer_heads) ** -0.5)
        attn_std = self._hidden_size**-0.5
        fc_std = (2 * self._hidden_size) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # nn.init.normal_(self.rgb_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.depth_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.inst_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.sub_fc[0].weight, std=self._hidden_size ** -0.5)

    def _t_forward(self, seq_embedding, attn_mask=None):
        ## Transformer
        if self.model_config.EVOENC.pre_ln:
            seq_embedding = self.pre_ln(seq_embedding)
            seq_embedding = self.pre_dropout(seq_embedding)
        seq_out = self.transformer(seq_embedding, attn_mask)
        if self.model_config.EVOENC.post_ln:
            seq_out = self.post_ln(seq_out)
            seq_out = self.post_dropout(seq_out)
        return seq_out

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Batch info
        # N = rnn_states.shape[0]
        # T = rnn_states.shape[1]

        # Embedding
        instruction_embedding = self.clip_encoder.encode_raw(observations)  # (N,L,D)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        sub_instruction_embedding = self.clip_encoder.encode_sub_instruction(
            observations
        )  # (N,L,D)
        mask_sub = (sub_instruction_embedding == 0).all(dim=2)
        depth_embedding, depth_embedding_seq = self.depth_encoder.encode_depth(
            observations
        )  # (N,D), (N,D,L)
        rgb_embedding, rgb_embedding_seq = self.clip_encoder.encode_image(
            observations,
        )  # (N,D), (N,D,L)

        # (T*N,D) -> (T*N,W,D)
        if "rgb_seq_features" not in observations:  # No dagger or eval
            # Cached features
            self.rgb_seq_features = rgb_embedding_seq
            self.rgb_features = rgb_embedding
            self.depth_seq_features = self.depth_encoder.get_depth_seq_features()
            self.inst_features = instruction_embedding
            self.sub_features = sub_instruction_embedding
        if self.model_config.EVOENC.prev_action:
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )

        # FC
        rgb_embedding_seq = torch.cat(
            [rgb_embedding.unsqueeze(2), rgb_embedding_seq], dim=2
        ).permute(
            0, 2, 1
        )  # (N, D, L+1)
        depth_embedding_seq = depth_embedding_seq.permute(0, 2, 1)
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        ## Construct input sequence
        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding.shape[0], -1, -1)
        ).clone()
        # Concat
        seq_embedding = torch.cat(
            [
                token_embeddings[:, 0:1, :],  # [RGB]
                rgb_embedding_seq,
                token_embeddings[:, 1:2, :],  # [DEP]
                depth_embedding_seq,
                token_embeddings[:, 2:3, :],  # [INS]
                instruction_embedding,
                token_embeddings[:, 3:4, :],  # [SUB]
                sub_instruction_embedding,
            ],
            dim=1,
        )
        # Extra embedding
        if self.pe_type == "position":
            seq_embedding = seq_embedding + self.positional_embedding.expand(
                (seq_embedding.shape[0], -1, -1)
            )
        elif self.pe_type == "token":
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = (
                seq_embedding
                + token_ids_embedding
                + self.positional_embedding.expand((seq_embedding.shape[0], -1, -1))
            )
        elif self.pe_type == "split_position":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = seq_embedding + split_position_embedding
        elif self.pe_type == "pt":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = (
                seq_embedding + split_position_embedding + token_ids_embedding
            )

        ## Attention mask, masking positions (empty words, empty subs)
        start_inst = 3 + self.depth_len + self.rgb_len
        start_sub = 4 + self.depth_len + self.rgb_len + self.instruction_len
        T_N = seq_embedding.shape[0]
        # (T*N, heads, L, L)
        attn_mask = torch.zeros(
            (T_N, self._transformer_heads, self.total_len, self.total_len), dtype=bool
        ).to(rgb_embedding.device)
        # (T*N, Linst) -> (T*N, heads, Linst, Linst)
        mask_inst = (
            mask_inst.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, self.total_len, -1)
        )
        mask_sub = (
            mask_sub.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, self.total_len, -1)
        )
        # Fill
        attn_mask[:, :, :, start_inst : start_sub - 1] = mask_inst
        attn_mask[:, :, :, start_sub:] = mask_sub
        attn_mask = attn_mask.reshape((-1, self.total_len, self.total_len))
        self.attn_mask = attn_mask

        # Transformer blocks
        seq_out = self._t_forward(seq_embedding, attn_mask)

        # Total feature, select
        rgb_feature = seq_out[:, 0, :].to(dtype=rnn_states.dtype)
        depth_feature = seq_out[:, self.rgb_len + 1, :].to(dtype=rnn_states.dtype)
        inst_feature = seq_out[:, self.rgb_len + self.depth_len + 2, :].to(
            dtype=rnn_states.dtype
        )
        sub_feature = seq_out[
            :, self.rgb_len + self.depth_len + self.instruction_len + 3, :
        ].to(dtype=rnn_states.dtype)

        if self.model_config.EVOENC.prev_action:
            rgb_feature = torch.cat([rgb_feature, prev_actions], dim=1)
            depth_feature = torch.cat([depth_feature, prev_actions], dim=1)
            inst_feature = torch.cat([inst_feature, prev_actions], dim=1)
            sub_feature = torch.cat([sub_feature, prev_actions], dim=1)

        # Decoder
        rnn_states_out = rnn_states.detach().clone()
        rgb_feature, rnn_states_out[:, 0 : self.s1] = self.action_rgb_decoder(
            rgb_feature,
            rnn_states[:, 0 : self.s1],
            masks,
        )
        depth_feature, rnn_states_out[:, self.s1 : self.s2] = self.action_depth_decoder(
            depth_feature,
            rnn_states[:, self.s1 : self.s2],
            masks,
        )
        inst_feature, rnn_states_out[:, self.s2 : self.s3] = self.action_inst_decoder(
            inst_feature,
            rnn_states[:, self.s2 : self.s3],
            masks,
        )
        sub_feature, rnn_states_out[:, self.s3 : self.s4] = self.action_sub_decoder(
            sub_feature,
            rnn_states[:, self.s3 : self.s4],
            masks,
        )
        if self.model_config.EVOENC.aggregate == "cat":
            total_feature = torch.cat(
                [rgb_feature, depth_feature, inst_feature, sub_feature], dim=1
            )
            total_feature = self.aggregate_ln(total_feature)
        elif self.model_config.EVOENC.aggregate == "add":
            total_feature = rgb_feature + depth_feature + inst_feature + sub_feature
            total_feature = self.aggregate_ln(total_feature)
        x = total_feature

        # AuxLosses
        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.float(),
                observations["progress"].float(),
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss.float(),
                self.model_config.PROGRESS_MONITOR.alpha,
            )
        return x, rnn_states_out

    def _feature_mask(self, feature, pad_mask=None):
        """True position in mask is masked"""
        ratio = self._masked_feature_ratio
        mask_embedding = 0
        if self.model_config.EVOENC.learnable_mask:
            mask_embedding = self.mask_embedding[:, : feature.shape[-1]]
        if pad_mask is not None:
            # for instruction
            N = feature.shape[0]
            lengths = pad_mask.logical_not().sum(dim=1)
            mask = torch.zeros_like(pad_mask, dtype=bool, device=feature.device)
            for i in range(N):
                num = int(lengths[i] * ratio)
                if num == 0 and lengths[i] >= 2:
                    num = 1
                pos = torch.randperm(lengths[i])[:num]
                mask[i][pos] = True
            masked_feature = feature.clone()
            masked_feature[pad_mask] = 0
            masked_feature[mask] = mask_embedding
            return masked_feature, mask
        else:
            # for vision
            N = feature.shape[0]
            L = feature.shape[1]
            num = int(L * ratio)
            pos = torch.randperm(L)[:num]
            mask = torch.zeros((N, L), dtype=bool, device=feature.device)
            mask[:, pos] = True
            masked_feature = feature.clone()
            # unable to use masked_feature[mask] = 0 with determinstic algorithms
            # masked_feature.masked_fill_(mask.unsqueeze(2).to(feature.device),0)
            masked_feature[mask] = mask_embedding
            return masked_feature, mask

    def stage1_forward(self, observations, positive=True):
        # Batch info
        # N = observations["rgb"].shape[0]

        # Embedding
        instruction_embedding = self.clip_encoder.encode_raw(observations)  # (N,L,D)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        sub_instruction_embedding = self.clip_encoder.encode_sub_instruction(
            observations
        )  # (N,L,D)
        mask_sub = (sub_instruction_embedding == 0).all(dim=2)
        depth_embedding, depth_embedding_seq = self.depth_encoder.encode_depth(
            observations
        )  # (N,D), (N,D,L)
        rgb_embedding, rgb_embedding_seq = self.clip_encoder.encode_image(
            observations,
        )  # (N,D), (N,D,L)
        rgb_embedding_seq = torch.cat(
            [rgb_embedding.unsqueeze(2), rgb_embedding_seq], dim=2
        ).permute(
            0, 2, 1
        )  # (N, D, L+1)
        depth_embedding_seq = depth_embedding_seq.permute(0, 2, 1)

        # Cached features
        self.rgb_seq_features = rgb_embedding_seq.detach().clone()
        self.depth_seq_features = (
            depth_embedding_seq.detach().clone()
        )  # self.depth_encoder.get_depth_seq_features()
        self.inst_features = instruction_embedding.detach().clone()
        self.sub_features = sub_instruction_embedding.detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        instruction_embedding, feature_mask_inst = self._feature_mask(
            instruction_embedding, pad_mask=mask_inst
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_instruction_embedding, pad_mask=mask_sub
        )
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_embedding_seq)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(
            depth_embedding_seq
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        #################################################
        # Losses initialization
        #################################################
        loss_rec = 0
        loss_mean = 0
        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding.shape[0], -1, -1)
        ).clone()

        #################################################
        # RGB part
        #################################################
        # Concat
        seq_rgb_embedding = torch.cat(
            [
                token_embeddings[:, 0:1, :],  # [RGB]
                rgb_embedding_seq,
            ],
            dim=1,
        )
        # Extra embedding
        if self.pe_type == "pt":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            split_position_embedding = torch.cat([a], dim=0).expand(
                (seq_rgb_embedding.shape[0], -1, -1)
            )
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            token_ids_embedding = torch.cat([a], dim=0).expand(
                (seq_rgb_embedding.shape[0], -1, -1)
            )
            seq_rgb_embedding = (
                seq_rgb_embedding + split_position_embedding + token_ids_embedding
            )

        # Transformer blocks
        seq_rgb_out = self._t_forward(seq_rgb_embedding)

        # split output sequence
        rgb_cls = seq_rgb_out[:, 0, :]
        rgb_out = seq_rgb_out[:, 1 : self.rgb_len + 1, :]
        # mask feature reconstruction
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb])
        loss_rec += F.mse_loss(rgb_rec, self.rgb_seq_features[feature_mask_rgb])
        # mean feature reconstruction
        rgb_mean_gt = self.rgb_seq_features.mean(dim=1)
        rgb_mean_rec = self.mean_rgb_reconstruction(rgb_cls)
        loss_mean += F.mse_loss(rgb_mean_rec, rgb_mean_gt)

        #################################################
        # Depth part
        #################################################
        # Concat
        seq_depth_embedding = torch.cat(
            [
                token_embeddings[:, 1:2, :],  # [DEP]
                depth_embedding_seq,
            ],
            dim=1,
        )
        # Extra embedding
        if self.pe_type == "pt":
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            split_position_embedding = torch.cat([b], dim=0).expand(
                (seq_depth_embedding.shape[0], -1, -1)
            )
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            token_ids_embedding = torch.cat([b], dim=0).expand(
                (seq_depth_embedding.shape[0], -1, -1)
            )
            seq_depth_embedding = (
                seq_depth_embedding + split_position_embedding + token_ids_embedding
            )
        # Transformer blocks
        seq_depth_out = self._t_forward(seq_depth_embedding)
        # split output sequence
        depth_cls = seq_depth_out[:, 0, :]
        depth_out = seq_depth_out[:, 1 : self.depth_len + 2 :, :]
        # mask feature reconstruction
        depth_rec = self.depth_reconstruction(depth_out[feature_mask_depth])
        loss_rec += F.mse_loss(depth_rec, self.depth_seq_features[feature_mask_depth])
        # mean feature reconstruction
        depth_mean_gt = self.depth_seq_features.mean(dim=1)
        depth_mean_rec = self.mean_depth_reconstruction(depth_cls)
        loss_mean += F.mse_loss(depth_mean_rec, depth_mean_gt)

        #################################################
        # Inst part
        #################################################
        seq_inst_embedding = torch.cat(
            [
                token_embeddings[:, 2:3, :],  # [INS]
                instruction_embedding,
            ],
            dim=1,
        )
        if self.pe_type == "pt":
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            split_position_embedding = torch.cat([c], dim=0).expand(
                (seq_inst_embedding.shape[0], -1, -1)
            )
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            token_ids_embedding = torch.cat([c], dim=0).expand(
                (seq_inst_embedding.shape[0], -1, -1)
            )
            seq_inst_embedding = (
                seq_inst_embedding + split_position_embedding + token_ids_embedding
            )
        start_inst = 1
        T_N = seq_inst_embedding.shape[0]
        # (T*N, heads, L, L)
        inst_len = 1 + self.instruction_len
        attn_mask = torch.zeros(
            (T_N, self._transformer_heads, inst_len, inst_len), dtype=bool
        ).to(rgb_embedding.device)
        # Fill, (T*N, Linst) -> (T*N, heads, Linst, Linst)
        attn_mask[:, :, :, start_inst:] = (
            mask_inst.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, inst_len, -1)
        )
        attn_mask = attn_mask.reshape((-1, inst_len, inst_len))
        # Transformer blocks
        seq_inst_out = self._t_forward(seq_inst_embedding, attn_mask)
        # split output sequence
        inst_cls = seq_inst_out[:, 0, :]
        inst_out = seq_inst_out[:, 1 : self.instruction_len + 1, :]
        # mask feature reconstruction
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst]) / COEF_REC_INST
        )
        # mean feature reconstruction
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            (~mask_inst).sum(dim=1, keepdim=True) + EPS
        )
        inst_mean_rec = self.mean_inst_reconstruction(inst_cls)
        loss_mean += F.mse_loss(inst_mean_rec, inst_mean_gt)

        #################################################
        # Sub part
        #################################################
        seq_sub_embedding = torch.cat(
            [
                token_embeddings[:, 3:4, :],  # [SUB]
                sub_instruction_embedding,
            ],
            dim=1,
        )
        if self.pe_type == "pt":
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([d], dim=0).expand(
                (seq_sub_embedding.shape[0], -1, -1)
            )
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([d], dim=0).expand(
                (seq_sub_embedding.shape[0], -1, -1)
            )
            seq_sub_embedding = (
                seq_sub_embedding + split_position_embedding + token_ids_embedding
            )
        start_sub = 1
        T_N = seq_sub_embedding.shape[0]
        # (T*N, heads, L, L)
        sub_len = 1 + self.sub_len
        attn_mask = torch.zeros(
            (T_N, self._transformer_heads, sub_len, sub_len), dtype=bool
        ).to(rgb_embedding.device)
        # Fill, (T*N, Linst) -> (T*N, heads, Linst, Linst)
        attn_mask[:, :, :, start_sub:] = (
            mask_sub.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, sub_len, -1)
        )
        attn_mask = attn_mask.reshape((-1, sub_len, sub_len))
        # Transformer blocks
        seq_sub_out = self._t_forward(seq_sub_embedding, attn_mask)
        # split output sequence
        sub_cls = seq_sub_out[:, 0, :]
        sub_out = seq_sub_out[:, 1:, :]
        # mask feature reconstruction
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub])
        loss_rec += F.mse_loss(sub_rec, self.sub_features[feature_mask_sub])
        # mean feature reconstruction
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            (~mask_sub).sum(dim=1, keepdim=True) + EPS
        )
        sub_mean_rec = self.mean_sub_reconstruction(sub_cls)
        loss_mean += F.mse_loss(sub_mean_rec, sub_mean_gt)

        return {
            "loss_rec": loss_rec,
            "loss_mean": loss_mean,
        }

    def stage2_forward(self, observations):
        #################################################
        # Losses initialization
        #################################################
        loss_rec = 0
        loss_mean = 0
        loss_align = 0
        align_gt = observations["inner_gt"]
        positive_idx = align_gt.bool()

        #################################################
        # Embeddings
        #################################################
        # Batch info
        # N = observations["rgb"].shape[0]

        # Embedding
        instruction_embedding = self.clip_encoder.encode_raw(observations)  # (N,L,D)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        sub_instruction_embedding = self.clip_encoder.encode_sub_instruction(
            observations
        )  # (N,L,D)
        mask_sub = (sub_instruction_embedding == 0).all(dim=2)
        depth_embedding, depth_embedding_seq = self.depth_encoder.encode_depth(
            observations
        )  # (N,D), (N,D,L)
        rgb_embedding, rgb_embedding_seq = self.clip_encoder.encode_image(
            observations,
        )  # (N,D), (N,D,L)
        rgb_embedding_seq = torch.cat(
            [rgb_embedding.unsqueeze(2), rgb_embedding_seq], dim=2
        ).permute(
            0, 2, 1
        )  # (N, D, L+1)
        depth_embedding_seq = depth_embedding_seq.permute(0, 2, 1)

        # Cached features, only positive samples used for reconstruction losses
        self.rgb_seq_features = rgb_embedding_seq[positive_idx].detach().clone()
        self.depth_seq_features = (
            depth_embedding_seq[positive_idx].detach().clone()
        )  # self.depth_encoder.get_depth_seq_features()
        self.inst_features = instruction_embedding[positive_idx].detach().clone()
        self.sub_features = sub_instruction_embedding[positive_idx].detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        instruction_embedding, feature_mask_inst = self._feature_mask(
            instruction_embedding, pad_mask=mask_inst
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_instruction_embedding, pad_mask=mask_sub
        )
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_embedding_seq)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(
            depth_embedding_seq
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        #################################################
        # Vision part
        #################################################
        # Token
        token_embeddings = self.token_embedding.expand((rgb_embedding.shape[0], -1, -1))
        # Concat
        seq_vision_embedding = torch.cat(
            [
                token_embeddings[:, 0:1, :],  # [RGB]
                rgb_embedding_seq,
                token_embeddings[:, 1:2, :],  # [DEP]
                depth_embedding_seq,
            ],
            dim=1,
        )
        # Extra embedding
        if self.pe_type == "pt":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            split_position_embedding = torch.cat([a, b], dim=0).expand(
                (seq_vision_embedding.shape[0], -1, -1)
            )
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            token_ids_embedding = torch.cat([a, b], dim=0).expand(
                (seq_vision_embedding.shape[0], -1, -1)
            )
            seq_vision_embedding = (
                seq_vision_embedding + split_position_embedding + token_ids_embedding
            )
        # Transformer blocks
        seq_vision_out = self._t_forward(seq_vision_embedding)
        # split output sequence
        rgb_cls = seq_vision_out[:, 0, :]
        depth_cls = seq_vision_out[:, self.rgb_len + 1, :]
        rgb_out = seq_vision_out[positive_idx, 1 : self.rgb_len + 1, :]
        depth_out = seq_vision_out[positive_idx, self.rgb_len + 2 :, :]
        # mask feature reconstruction, only positive samples are involved
        feature_mask_rgb = feature_mask_rgb[positive_idx]
        feature_mask_depth = feature_mask_depth[positive_idx]
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb])
        depth_rec = self.depth_reconstruction(depth_out[feature_mask_depth])
        loss_rec += F.mse_loss(rgb_rec, self.rgb_seq_features[feature_mask_rgb])
        loss_rec += F.mse_loss(depth_rec, self.depth_seq_features[feature_mask_depth])
        # mean feature reconstruction, only positive samples are involved
        rgb_mean_gt = self.rgb_seq_features.mean(dim=1)
        rgb_mean_rec = self.mean_rgb_reconstruction(rgb_cls[positive_idx])
        loss_mean += F.mse_loss(rgb_mean_rec, rgb_mean_gt)
        depth_mean_gt = self.depth_seq_features.mean(dim=1)
        depth_mean_rec = self.mean_depth_reconstruction(depth_cls[positive_idx])
        loss_mean += F.mse_loss(depth_mean_rec, depth_mean_gt)
        # inner alignment, all samples are involved
        rgb_depth_cls = torch.cat([rgb_cls, depth_cls], dim=1)
        align_pre_v = self.inner_alignment(rgb_depth_cls)
        loss_align += F.binary_cross_entropy(
            torch.sigmoid(align_pre_v.squeeze()), align_gt.float()
        )

        #################################################
        # Language part
        #################################################
        seq_language_embedding = torch.cat(
            [
                token_embeddings[:, 2:3, :],  # [INS]
                instruction_embedding,
                token_embeddings[:, 3:4, :],  # [SUB]
                sub_instruction_embedding,
            ],
            dim=1,
        )
        if self.pe_type == "pt":
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([c, d], dim=0).expand(
                (seq_language_embedding.shape[0], -1, -1)
            )
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([c, d], dim=0).expand(
                (seq_language_embedding.shape[0], -1, -1)
            )
            seq_language_embedding = (
                seq_language_embedding + split_position_embedding + token_ids_embedding
            )
        start_inst = 1
        start_sub = 2 + self.instruction_len
        T_N = seq_language_embedding.shape[0]
        # (T*N, heads, L, L)
        language_len = 2 + self.instruction_len + self.sub_len
        attn_mask = torch.zeros(
            (T_N, self._transformer_heads, language_len, language_len), dtype=bool
        ).to(rgb_embedding.device)
        # Fill, (T*N, Linst) -> (T*N, heads, Linst, Linst)
        attn_mask[:, :, :, start_inst : start_sub - 1] = (
            mask_inst.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, language_len, -1)
        )
        attn_mask[:, :, :, start_sub:] = (
            mask_sub.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, language_len, -1)
        )
        attn_mask = attn_mask.reshape((-1, language_len, language_len))
        # Transformer blocks
        seq_language_out = self._t_forward(seq_language_embedding, attn_mask)
        # split output sequence
        inst_cls = seq_language_out[:, 0, :]
        sub_cls = seq_language_out[:, self.instruction_len + 1, :]
        inst_out = seq_language_out[positive_idx, 1 : self.instruction_len + 1, :]
        sub_out = seq_language_out[positive_idx, self.instruction_len + 2 :, :]
        # mask feature reconstruction, only positive samples are involved
        feature_mask_inst = feature_mask_inst[positive_idx]
        feature_mask_sub = feature_mask_sub[positive_idx]
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst])
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst]) / COEF_REC_INST
        )
        loss_rec += F.mse_loss(sub_rec, self.sub_features[feature_mask_sub])
        # mean feature reconstruction
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            (~(mask_inst[positive_idx])).sum(dim=1, keepdim=True) + EPS
        )
        inst_mean_rec = self.mean_inst_reconstruction(inst_cls[positive_idx])
        loss_mean += F.mse_loss(inst_mean_rec, inst_mean_gt)
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            (~(mask_sub[positive_idx])).sum(dim=1, keepdim=True) + EPS
        )
        sub_mean_rec = self.mean_sub_reconstruction(sub_cls[positive_idx])
        loss_mean += F.mse_loss(sub_mean_rec, sub_mean_gt)
        # inner alignment
        inst_sub_cls = torch.cat([inst_cls, sub_cls], dim=1)
        align_pre_l = self.inner_alignment(inst_sub_cls)
        loss_align += F.binary_cross_entropy(
            torch.sigmoid(align_pre_l.squeeze()), align_gt.float()
        )

        return {
            "loss_rec": loss_rec if positive_idx.sum() > 0 else 0,
            "loss_mean": loss_mean if positive_idx.sum() > 0 else 0,
            "loss_align": loss_align,
        }, {
            "align_gt_v": align_gt,
            "align_pre_v": align_pre_v,
            "align_gt_l": align_gt,
            "align_pre_l": align_pre_l,
        }

    def stage3_forward(self, observations):
        #################################################
        # Losses initialization
        #################################################
        loss_rec = 0
        loss_mean = 0
        loss_inner = 0
        loss_outer = 0
        inner_gt = observations["inner_gt"]
        outer_gt = observations["outer_gt"]
        positive_idx = outer_gt.bool()

        #################################################
        # Embeddings
        #################################################
        # Batch info
        # N = observations["rgb"].shape[0]

        # Embedding
        instruction_embedding = self.clip_encoder.encode_raw(observations)  # (N,L,D)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        sub_instruction_embedding = self.clip_encoder.encode_sub_instruction(
            observations
        )  # (N,L,D)
        mask_sub = (sub_instruction_embedding == 0).all(dim=2)
        depth_embedding, depth_embedding_seq = self.depth_encoder.encode_depth(
            observations
        )  # (N,D), (N,D,L)
        rgb_embedding, rgb_embedding_seq = self.clip_encoder.encode_image(
            observations,
        )  # (N,D), (N,D,L)
        rgb_embedding_seq = torch.cat(
            [rgb_embedding.unsqueeze(2), rgb_embedding_seq], dim=2
        ).permute(
            0, 2, 1
        )  # (N, D, L+1)
        depth_embedding_seq = depth_embedding_seq.permute(0, 2, 1)

        # Cached features, only positive samples used for reconstruction losses
        self.rgb_seq_features = rgb_embedding_seq[positive_idx].detach().clone()
        self.depth_seq_features = (
            depth_embedding_seq[positive_idx].detach().clone()
        )  # self.depth_encoder.get_depth_seq_features()
        self.inst_features = instruction_embedding[positive_idx].detach().clone()
        self.sub_features = sub_instruction_embedding[positive_idx].detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        instruction_embedding, feature_mask_inst = self._feature_mask(
            instruction_embedding, pad_mask=mask_inst
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_instruction_embedding, pad_mask=mask_sub
        )
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_embedding_seq)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(
            depth_embedding_seq
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        ## Construct input sequence
        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding.shape[0], -1, -1)
        ).clone()
        # Concat
        seq_embedding = torch.cat(
            [
                token_embeddings[:, 0:1, :],  # [RGB]
                rgb_embedding_seq,
                token_embeddings[:, 1:2, :],  # [DEP]
                depth_embedding_seq,
                token_embeddings[:, 2:3, :],  # [INS]
                instruction_embedding,
                token_embeddings[:, 3:4, :],  # [SUB]
                sub_instruction_embedding,
            ],
            dim=1,
        )
        # Extra embedding
        if self.pe_type == "position":
            seq_embedding = seq_embedding + self.positional_embedding.expand(
                (seq_embedding.shape[0], -1, -1)
            )
        elif self.pe_type == "token":
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = seq_embedding + token_ids_embedding
        elif self.pe_type == "split_position":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = seq_embedding + split_position_embedding
        elif self.pe_type == "pt":
            a = self.positional_embedding[0 : self.rgb_len + 1, :]
            b = self.positional_embedding[0 : self.depth_len + 1, :]
            c = self.positional_embedding[0 : self.instruction_len + 1, :]
            d = self.positional_embedding[0 : self.sub_len + 1, :]
            split_position_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            a = self.type_embedding[0:1, :].repeat((self.rgb_len + 1, 1))
            b = self.type_embedding[1:2, :].repeat((self.depth_len + 1, 1))
            c = self.type_embedding[2:3, :].repeat((self.instruction_len + 1, 1))
            d = self.type_embedding[3:4, :].repeat((self.sub_len + 1, 1))
            token_ids_embedding = torch.cat([a, b, c, d], dim=0).expand(
                (seq_embedding.shape[0], -1, -1)
            )
            seq_embedding = (
                seq_embedding + split_position_embedding + token_ids_embedding
            )

        ## Attention mask, masking positions (empty words, empty subs)
        start_inst = 3 + self.depth_len + self.rgb_len
        start_sub = 4 + self.depth_len + self.rgb_len + self.instruction_len
        T_N = seq_embedding.shape[0]
        # (T*N, heads, L, L)
        attn_mask = torch.zeros(
            (T_N, self._transformer_heads, self.total_len, self.total_len), dtype=bool
        ).to(rgb_embedding.device)
        # Fill
        attn_mask[:, :, :, start_inst : start_sub - 1] = (
            mask_inst.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, self.total_len, -1)
        )
        attn_mask[:, :, :, start_sub:] = (
            mask_sub.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self._transformer_heads, self.total_len, -1)
        )
        attn_mask = attn_mask.reshape((-1, self.total_len, self.total_len))
        self.attn_mask = attn_mask

        # Transformer blocks
        seq_out = self._t_forward(seq_embedding, attn_mask)

        # Total feature, select
        rgb_cls = seq_out[:, 0, :]
        depth_cls = seq_out[:, self.rgb_len + 1, :]
        inst_cls = seq_out[:, start_inst - 1, :]
        sub_cls = seq_out[:, start_sub - 1, :]
        rgb_out = seq_out[
            positive_idx,
            1 : self.rgb_len + 1,
        ]
        depth_out = seq_out[
            positive_idx, self.rgb_len + 2 : self.rgb_len + self.depth_len + 2, :
        ]
        inst_out = seq_out[positive_idx, start_inst : start_sub - 1, :]
        sub_out = seq_out[positive_idx, start_sub:, :]

        # mask feature reconstruction, only positive samples are involved
        feature_mask_rgb = feature_mask_rgb[positive_idx]
        feature_mask_depth = feature_mask_depth[positive_idx]
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb])
        depth_rec = self.depth_reconstruction(depth_out[feature_mask_depth])
        loss_rec += F.mse_loss(rgb_rec, self.rgb_seq_features[feature_mask_rgb])
        loss_rec += F.mse_loss(depth_rec, self.depth_seq_features[feature_mask_depth])
        feature_mask_inst = feature_mask_inst[positive_idx]
        feature_mask_sub = feature_mask_sub[positive_idx]
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst])
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst]) / COEF_REC_INST
        )
        loss_rec += F.mse_loss(sub_rec, self.sub_features[feature_mask_sub])

        # mean feature reconstruction, only positive samples are involved
        rgb_mean_gt = self.rgb_seq_features.mean(dim=1)
        rgb_mean_rec = self.mean_rgb_reconstruction(rgb_cls[positive_idx])
        loss_mean += F.mse_loss(rgb_mean_rec, rgb_mean_gt)
        depth_mean_gt = self.depth_seq_features.mean(dim=1)
        depth_mean_rec = self.mean_depth_reconstruction(depth_cls[positive_idx])
        loss_mean += F.mse_loss(depth_mean_rec, depth_mean_gt)
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            (~(mask_inst[positive_idx])).sum(dim=1, keepdim=True) + EPS
        )
        inst_mean_rec = self.mean_inst_reconstruction(inst_cls[positive_idx])
        loss_mean += F.mse_loss(inst_mean_rec, inst_mean_gt)
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            (~(mask_sub[positive_idx])).sum(dim=1, keepdim=True) + EPS
        )
        sub_mean_rec = self.mean_sub_reconstruction(sub_cls[positive_idx])
        loss_mean += F.mse_loss(sub_mean_rec, sub_mean_gt)

        # inner alignment, all samples are involved
        rgb_depth_cls = torch.cat([rgb_cls, depth_cls], dim=1)
        align_pre_v = self.inner_alignment(rgb_depth_cls)
        loss_inner += F.binary_cross_entropy(
            torch.sigmoid(align_pre_v.squeeze()), inner_gt.float()
        )
        inst_sub_cls = torch.cat([inst_cls, sub_cls], dim=1)
        align_pre_l = self.inner_alignment(inst_sub_cls)
        loss_inner += F.binary_cross_entropy(
            torch.sigmoid(align_pre_l.squeeze()), inner_gt.float()
        )

        # outer alignment, all samples are involved
        rgb_depth_inst_sub_cls = torch.cat(
            [rgb_cls, depth_cls, inst_cls, sub_cls], dim=1
        )
        align_pre = self.outer_alignment(rgb_depth_inst_sub_cls)
        loss_outer += F.binary_cross_entropy(
            torch.sigmoid(align_pre.squeeze()), outer_gt.float()
        )

        return {
            "loss_rec": loss_rec if positive_idx.sum() > 0 else 0,
            "loss_mean": loss_mean if positive_idx.sum() > 0 else 0,
            "loss_inner": loss_inner,
            "loss_outer": loss_outer,
        }, {
            "inner_gt_v": inner_gt,
            "inner_pre_v": align_pre_v,
            "inner_gt_l": inner_gt,
            "inner_pre_l": align_pre_l,
            "outer_gt": outer_gt,
            "outer_pre": align_pre,
        }


if __name__ == "__main__":
    device = torch.device("cuda:6")
    a = torch.randn((10, 77, 512))
    a = a.to(device)
    model = EENet(None, None, None)
    model.to(device)
    b = model.test(a)
    print(b)
