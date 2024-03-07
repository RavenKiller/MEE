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
from evoenc.common.constants import PAD_IDX, SUB_PAD_IDX
from transformers import CLIPVisionModel, BertModel, RobertaModel, AutoModel, ViTMAEModel
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention
from sentence_transformers import SentenceTransformer

# from vlnce_baselines.models.encoders import resnet_encoders
from evoenc.models.encoders.transformer_encoder import Transformer, LayerNorm
from evoenc.models.encoders.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from evoenc.models.policy import ILPolicy

RGB = 0
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


def sub_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
def inf_mask(attention_mask):
    dtype = torch.float32
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class ReconstructionLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, out_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        # x = self.activation(x)
        return x


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
        excludes: list = ["clip_encoder", "tac_encoder","bert_encoder","sbert_encoder"],
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
        if "mae" in config.MODEL.CLIP.model_name:
            self.clip_encoder = ViTMAEModel.from_pretrained(
                config.MODEL.CLIP.model_name
            )
        else:
            self.clip_encoder = CLIPVisionModel.from_pretrained(
                config.MODEL.CLIP.model_name
            )
        # Init the TAC depth encoder
        self.tac_encoder = CLIPVisionModel.from_pretrained(config.MODEL.TAC.model_name)
        # Init the BERT text encoder
        self.roberta_encoder = RobertaModel.from_pretrained(
            config.MODEL.BERT.model_name
        )
        # Init the SentenceBERT sub encoder
        # self.sbert_encoder = SentenceTransformer(config.MODEL.SBERT.model_name)
        self.sbert_encoder = AutoModel.from_pretrained(config.MODEL.SBERT.model_name)
        # Set trainable
        for param in self.clip_encoder.parameters():
            param.requires_grad_(config.MODEL.CLIP.trainable)
        for param in self.tac_encoder.parameters():
            param.requires_grad_(config.MODEL.TAC.trainable)
        for param in self.roberta_encoder.parameters():
            param.requires_grad_(config.MODEL.BERT.trainable)
        for param in self.sbert_encoder.parameters():
            param.requires_grad_(config.MODEL.SBERT.trainable)

        self.rgb_fc = nn.Sequential(
            # nn.LayerNorm(model_config.CLIP.hidden_size),
            nn.Dropout(p=model_config.EVOENC.dropout),
        )
        self.depth_fc = nn.Sequential(
            # nn.LayerNorm(model_config.TAC.hidden_size),
            nn.Dropout(p=model_config.EVOENC.dropout),
        )
        self.inst_fc = nn.Sequential(
            # nn.LayerNorm(model_config.BERT.hidden_size),
            nn.Dropout(p=model_config.EVOENC.dropout),
        )
        self.sub_fc = nn.Sequential(
            # nn.LayerNorm(model_config.SBERT.hidden_size),
            nn.Dropout(p=model_config.EVOENC.dropout),
        )
        self.window_size = model_config.EVOENC.window_size
        self.rgb_len = model_config.EVOENC.rgb_len
        self.depth_len = model_config.EVOENC.depth_len
        self.instruction_len = model_config.EVOENC.instruction_len
        self.sub_len = model_config.EVOENC.sub_len
        self.pe_type = model_config.EVOENC.pe_type
        self.storage = None

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

        if model_config.EVOENC.pre_ln:
            self.pre_ln = nn.LayerNorm(self._hidden_size)
            self.pre_dropout = nn.Dropout(p=model_config.EVOENC.pre_dropout)
        bert_config = BertConfig(
            hidden_size=self._hidden_size,
            num_hidden_layers=self._transformer_layers,
            num_attention_heads=self._transformer_heads,
            hidden_dropout_prob=self._inner_dropout,
            attention_probs_dropout_prob=self._inner_dropout,
            position_embedding_type="relative_key_query",
            type_vocab_size=4,
        )
        self.transformer = BertModel(bert_config)
        del self.transformer.embeddings.word_embeddings  # useless word embeddings
        if model_config.EVOENC.post_ln:
            self.post_ln = nn.LayerNorm(self._hidden_size)
            self.post_dropout = nn.Dropout(p=model_config.EVOENC.post_dropout)

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
        self.fuse_layer = nn.Sequential(
            nn.Linear(model_config.STATE_ENCODER.hidden_size*4, self._hidden_size),
            nn.GELU(),
        )
        
        action_attn_config = BertConfig(
            hidden_size=self._hidden_size,
            num_attention_heads=self._transformer_heads,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        self.action_attn_rgb = BertAttention(action_attn_config)
        self.action_attn_depth = BertAttention(action_attn_config)
        self.action_attn_inst = BertAttention(action_attn_config)
        self.action_attn_sub = BertAttention(action_attn_config)

        final_input_size = rnn_input_size*4+self._hidden_size*5
        if model_config.EVOENC.prev_action:
            final_input_size += self.prev_action_embedding.embedding_dim
        self.action_decoder = build_rnn_state_encoder(
            input_size=final_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type_action,
            num_layers=model_config.STATE_ENCODER.num_layers_action,
            dropout=dropout_ratio_rnn,
        )
        self._num_recurrent_layers = self.action_rgb_decoder.num_recurrent_layers * 5
        self.s1 = self.action_rgb_decoder.num_recurrent_layers
        self.s2 = self.action_rgb_decoder.num_recurrent_layers * 2
        self.s3 = self.action_rgb_decoder.num_recurrent_layers * 3
        self.s4 = self.action_rgb_decoder.num_recurrent_layers * 4
        self.s5 = self.action_rgb_decoder.num_recurrent_layers * 5

        self.rgb_features = None
        self.depth_features = None
        self.inst_features = None
        self.sub_features = None

        self.attn_mask = None

        # Init the progress monitor
        self.progress_monitor = nn.Linear(self._output_size, 1)
        if model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
            nn.init.constant_(self.progress_monitor.bias, 0)

        #############################################################
        # For pretrain, define some heads
        #############################################################
        if model_config.EVOENC.learnable_mask:
            self.mask_embedding = nn.Parameter(torch.empty(1, 768))
        if self.config.PRETRAIN.stage != "NONE":
            # masked feature reconstruction
            self.rgb_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.CLIP.hidden_size
            )
            self.depth_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.TAC.hidden_size
            )
            self.inst_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.BERT.hidden_size
            )
            self.sub_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.SBERT.hidden_size
            )
            # mean feature reconstruction
            self.mean_rgb_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.CLIP.hidden_size
            )
            self.mean_depth_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.TAC.hidden_size
            )
            self.mean_inst_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.BERT.hidden_size
            )
            self.mean_sub_reconstruction = ReconstructionLayer(
                self._hidden_size, model_config.SBERT.hidden_size
            )
            # feature type prediction
            # self.type_prediction = nn.Linear(self._hidden_size, 4)
            # feature alignment
            self.inner_alignment_v = nn.Linear(self._hidden_size * 2, 1)
            self.inner_alignment_l = nn.Linear(self._hidden_size * 2, 1)
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

    def encode_rgb(self, observations):
        if "rgb_seq_features" in observations:  # [B, L_rgb, D]
            rgb_seq_features = observations["rgb_seq_features"]
        else:
            if "mae" in self.config.MODEL.CLIP.model_name:
                rgb_observations = observations["rgb"]
                rgb_seq_features = self.clip_encoder(
                    pixel_values=rgb_observations
                ).last_hidden_state
            else:
                rgb_observations = observations["rgb"]
                rgb_seq_features = self.clip_encoder(
                    pixel_values=rgb_observations, output_hidden_states=True
                ).hidden_states[-2]
        return rgb_seq_features

    def encode_depth(self, observations):
        if "depth_seq_features" in observations:  # [B, L_depth, D]
            depth_seq_features = observations["depth_seq_features"]
        else:
            depth_observations = observations["depth"]
            depth_seq_features = self.tac_encoder(
                pixel_values=depth_observations, output_hidden_states=True
            ).hidden_states[-2]
        return depth_seq_features

    def encode_inst(self, observations):
        # In roberta, attention mask=1 is words, =0 is padding
        if "inst_seq_features" in observations:
            inst_seq_features = observations["inst_seq_features"]  # [B, L_isnt, D]
        else:
            inst_observations = observations.get("instruction", None)
            inst_mask = observations.get("instruction_mask", None)
            if inst_observations is None:
                inst_observations = observations.get("text", None)
                inst_mask = observations.get("text_mask", None)
            if inst_mask is None:
                inst_mask = (inst_observations!=PAD_IDX)
            inst_observations = inst_observations.long()
            inst_seq_features = self.roberta_encoder(
                input_ids=inst_observations, attention_mask=inst_mask
            ).last_hidden_state
        return inst_seq_features, inst_mask

    def encode_sub(self, observations):
        # In sentence bert, attention mask=1 is words, =0 is padding
        if "sub_seq_features" in observations:
            sub_seq_features = observations["sub_seq_features"]  # [B, L_sub, D]
        else:
            sub_observations = observations.get("sub_instruction", None)
            sub_mask = observations.get("sub_instruction_mask", None)
            if sub_observations is None:
                sub_observations = observations.get("sub", None)
                sub_mask = observations.get("sub_mask", None)
            if sub_mask is None:
                sub_mask = (sub_observations!=SUB_PAD_IDX)
            sub_observations = sub_observations.long()
            B, S, L = sub_observations.shape
            model_output = self.sbert_encoder(
                input_ids=sub_observations.view((B * S, L)),
                attention_mask=sub_mask.view((B * S, L)),
            )
            sub_seq_features = sub_mean_pooling(
                model_output, sub_mask.view((B * S, L))
            ).view((B, S, -1))
        return sub_seq_features, sub_mask.any(dim=-1)

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
        # proj_std = (self._hidden_size**-0.5) * ((2 * self._transformer_heads) ** -0.5)
        # attn_std = self._hidden_size**-0.5
        # fc_std = (2 * self._hidden_size) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # nn.init.normal_(self.rgb_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.depth_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.inst_fc[0].weight, std=self._hidden_size ** -0.5)
        # nn.init.normal_(self.sub_fc[0].weight, std=self._hidden_size ** -0.5)

    def _t_forward(self, seq_embedding, attention_mask=None, token_type_ids=None):
        ## Transformer
        if self.model_config.EVOENC.pre_ln:
            seq_embedding = self.pre_ln(seq_embedding)
            seq_embedding = self.pre_dropout(seq_embedding)
        seq_out = self.transformer(
            inputs_embeds=seq_embedding,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
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
        #################################################
        # Embeddings
        #################################################
        with torch.no_grad():
            rgb_seq_features = self.encode_rgb(observations)
            depth_seq_features = self.encode_depth(observations)
            inst_seq_features, inst_mask = self.encode_inst(observations)
            sub_seq_features, sub_mask = self.encode_sub(observations)
        device = rgb_seq_features.device

        if "rgb_seq_features" not in observations:  # No dagger or eval
            # Cached features
            self.rgb_seq_features = rgb_seq_features
            self.depth_seq_features = depth_seq_features
            self.inst_seq_features = inst_seq_features
            self.sub_seq_features = sub_seq_features
        if self.model_config.EVOENC.prev_action:
            prev_actions = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )

        # FC
        rgb_embedding_seq = self.rgb_fc(rgb_seq_features)
        depth_embedding_seq = self.depth_fc(depth_seq_features)
        instruction_embedding = self.inst_fc(inst_seq_features)
        sub_instruction_embedding = self.sub_fc(sub_seq_features)

        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding_seq.shape[0], -1, -1)
        )

        #################################################
        # All modalities
        #################################################
        ## Construct input sequence
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

        # Modality type ids
        B, L_rgb, D_rgb = rgb_embedding_seq.shape
        B, L_depth, D_depth = depth_embedding_seq.shape
        B, L_inst, D_inst = instruction_embedding.shape
        B, L_sub, D_sub = sub_instruction_embedding.shape

        all_type_ids = torch.cat(
            [
                torch.ones((B, L_rgb + 1), dtype=int, device=device) * RGB,
                torch.ones((B, L_depth + 1), dtype=int, device=device) * DEP,
                torch.ones((B, L_inst + 1), dtype=int, device=device) * INS,
                torch.ones((B, L_sub + 1), dtype=int, device=device) * SUB,
            ],
            dim=1,
        )
        # Attention mask
        attn_mask = torch.ones(
            (B, L_rgb + L_depth + L_inst + L_sub + 4), dtype=int, device=device
        )
        attn_mask[:, L_rgb + L_depth + 3 : L_rgb + L_depth + L_inst + 3] = inst_mask
        attn_mask[:, L_rgb + L_depth + L_inst + 4 :] = sub_mask

        # Transformer blocks
        seq_out = self._t_forward(
            seq_embedding, attention_mask=attn_mask, token_type_ids=all_type_ids
        )

        # Total feature, select
        rgb_cls = seq_out[:, 0, :].to(dtype=rnn_states.dtype)
        depth_cls = seq_out[:, L_rgb + 1, :].to(dtype=rnn_states.dtype)
        inst_cls = seq_out[:, L_rgb + L_depth + 2, :].to(dtype=rnn_states.dtype)
        sub_cls = seq_out[:, L_rgb + L_depth + L_inst + 3, :].to(dtype=rnn_states.dtype)
        # rgb_out = seq_out[:,1 : L_rgb + 1, ]
        # depth_out = seq_out[:, L_rgb + 2 : L_rgb + L_depth + 2, :]
        # inst_out = seq_out[:, L_rgb + L_depth + 3 : L_rgb + L_depth + L_inst + 3, :]
        # sub_out = seq_out[:, L_rgb + L_depth + L_inst + 4 :, :]

        if self.model_config.EVOENC.prev_action:
            rgb_cls = torch.cat([rgb_cls, prev_actions], dim=1)
            depth_cls = torch.cat([depth_cls, prev_actions], dim=1)
            inst_cls = torch.cat([inst_cls, prev_actions], dim=1)
            sub_cls = torch.cat([sub_cls, prev_actions], dim=1)

        #################################################
        # Decoder
        #################################################
        rnn_states_out = rnn_states.detach().clone()
        rgb_feature, rnn_states_out[:, 0 : self.s1] = self.action_rgb_decoder(
            rgb_cls,
            rnn_states[:, 0 : self.s1],
            masks,
        )
        depth_feature, rnn_states_out[:, self.s1 : self.s2] = self.action_depth_decoder(
            depth_cls,
            rnn_states[:, self.s1 : self.s2],
            masks,
        )
        inst_feature, rnn_states_out[:, self.s2 : self.s3] = self.action_inst_decoder(
            inst_cls,
            rnn_states[:, self.s2 : self.s3],
            masks,
        )
        sub_feature, rnn_states_out[:, self.s3 : self.s4] = self.action_sub_decoder(
            sub_cls,
            rnn_states[:, self.s3 : self.s4],
            masks,
        )
        fused_feature = self.fuse_layer(torch.cat((rgb_feature, depth_feature, inst_feature, sub_feature), dim=1))

        # Decoder attention
        attn_rgb_feature = self.action_attn_rgb(hidden_states=fused_feature[:,None,:], encoder_hidden_states=rgb_seq_features)[0].squeeze(1)
        attn_depth_feature = self.action_attn_depth(hidden_states=fused_feature[:,None,:], encoder_hidden_states=depth_seq_features)[0].squeeze(1)
        attn_inst_feature = self.action_attn_inst(hidden_states=fused_feature[:,None,:], encoder_hidden_states=inst_seq_features, encoder_attention_mask=inf_mask(inst_mask))[0].squeeze(1)
        attn_sub_feature = self.action_attn_sub(hidden_states=fused_feature[:,None,:], encoder_hidden_states=sub_seq_features,encoder_attention_mask=inf_mask(sub_mask))[0].squeeze(1)

        total_feature = torch.cat((
            rgb_cls,
            depth_cls,
            inst_cls,
            sub_cls,
            fused_feature,
            attn_rgb_feature,
            attn_depth_feature,
            attn_inst_feature,
            attn_sub_feature
        ), dim=1)
        if self.model_config.EVOENC.prev_action:
            total_feature = torch.cat((total_feature, prev_actions), dim=1)
        
        x, rnn_states_out[:, self.s4 : self.s5] = self.action_decoder(
            total_feature,
            rnn_states[:, self.s4 : self.s5],
            masks,
        )

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
        """True position IS NOT masked, False position IS masked"""
        ratio = self._masked_feature_ratio
        mask_embedding = 0
        if self.model_config.EVOENC.learnable_mask:
            mask_embedding = self.mask_embedding[:, : feature.shape[-1]]
        if pad_mask is not None:
            # for instruction
            N = feature.shape[0]
            lengths = pad_mask.sum(dim=1)
            mask = torch.ones_like(pad_mask, dtype=bool, device=feature.device)
            for i in range(N):
                num = int(lengths[i] * ratio)
                if num == 0 and lengths[i] >= 2:
                    num = 1
                pos = torch.randperm(lengths[i])[:num]
                mask[i][pos] = False
            masked_feature = feature.clone()
            masked_feature[pad_mask.logical_not()] = 0
            masked_feature[mask.logical_not()] = mask_embedding
            return masked_feature, mask
        else:
            # for vision
            N = feature.shape[0]
            L = feature.shape[1]
            num = int(L * ratio)
            pos = torch.randperm(L)[:num]
            mask = torch.ones((N, L), dtype=bool, device=feature.device)
            mask[:, pos] = False
            masked_feature = feature.clone()
            # unable to use masked_feature[mask] = 0 with determinstic algorithms
            # masked_feature.masked_fill_(mask.unsqueeze(2).to(feature.device),0)
            masked_feature[mask.logical_not()] = mask_embedding
            return masked_feature, mask

    def stage1_forward(self, observations, positive=True):
        #################################################
        # Losses initialization
        #################################################
        loss_rec = 0
        loss_mean = 0

        #################################################
        # Embeddings
        #################################################
        # Batch info
        B = observations["rgb"].shape[0]
        device = observations["rgb"].device

        # Embedding
        rgb_seq_features = self.encode_rgb(observations)
        depth_seq_features = self.encode_depth(observations)
        inst_seq_features, inst_mask = self.encode_inst(observations)
        sub_seq_features, sub_mask = self.encode_sub(observations)

        # Cached features
        self.rgb_seq_features = rgb_seq_features.detach().clone()
        self.depth_seq_features = depth_seq_features.detach().clone()
        self.inst_features = inst_seq_features.detach().clone()
        self.sub_features = sub_seq_features.detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_seq_features)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(depth_seq_features)
        instruction_embedding, feature_mask_inst = self._feature_mask(
            inst_seq_features, pad_mask=inst_mask
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_seq_features, pad_mask=sub_mask
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding_seq.shape[0], -1, -1)
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
        # Modality type ids
        B, L, D = seq_rgb_embedding.shape
        rgb_type_ids = torch.ones((B, L), dtype=int, device=device) * RGB

        # Transformer blocks
        seq_rgb_out = self._t_forward(seq_rgb_embedding, token_type_ids=rgb_type_ids)

        # split output sequence
        rgb_cls = seq_rgb_out[:, 0, :]
        rgb_out = seq_rgb_out[:, 1:, :]
        # mask feature reconstruction
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb.logical_not()])
        loss_rec += F.mse_loss(
            rgb_rec, self.rgb_seq_features[feature_mask_rgb.logical_not()]
        )
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
        # Modality type ids
        B, L, D = seq_depth_embedding.shape
        depth_type_ids = torch.ones((B, L), dtype=int, device=device) * DEP

        # Transformer blocks
        seq_depth_out = self._t_forward(
            seq_depth_embedding, token_type_ids=depth_type_ids
        )

        # split output sequence
        depth_cls = seq_depth_out[:, 0, :]
        depth_out = seq_depth_out[:, 1:, :]
        # mask feature reconstruction
        depth_rec = self.depth_reconstruction(
            depth_out[feature_mask_depth.logical_not()]
        )
        loss_rec += F.mse_loss(
            depth_rec, self.depth_seq_features[feature_mask_depth.logical_not()]
        )
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
        # Modality type ids
        B, L, D = seq_inst_embedding.shape
        inst_type_ids = torch.ones((B, L), dtype=int, device=device) * INS

        # Attention mask
        attn_mask = torch.ones((B, L), dtype=int, device=device)
        attn_mask[:, 1:] = inst_mask

        # Transformer blocks
        seq_inst_out = self._t_forward(
            seq_inst_embedding, attention_mask=attn_mask, token_type_ids=inst_type_ids
        )

        # split output sequence
        inst_cls = seq_inst_out[:, 0, :]
        inst_out = seq_inst_out[:, 1:, :]
        # mask feature reconstruction
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst.logical_not()])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst.logical_not()])
            / COEF_REC_INST
        )
        # mean feature reconstruction
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            inst_mask.sum(dim=1, keepdim=True) + EPS
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
        # Modality type ids
        B, L, D = seq_sub_embedding.shape
        sub_type_ids = torch.ones((B, L), dtype=int, device=device) * SUB

        # Attention mask
        attn_mask = torch.ones((B, L), dtype=int, device=device)
        attn_mask[:, 1:] = sub_mask

        # Transformer blocks
        seq_sub_out = self._t_forward(
            seq_sub_embedding, attention_mask=attn_mask, token_type_ids=sub_type_ids
        )

        # split output sequence
        sub_cls = seq_sub_out[:, 0, :]
        sub_out = seq_sub_out[:, 1:, :]
        # mask feature reconstruction
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub.logical_not()])
        loss_rec += F.mse_loss(
            sub_rec, self.sub_features[feature_mask_sub.logical_not()]
        )
        # mean feature reconstruction
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            sub_mask.sum(dim=1, keepdim=True) + EPS
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
        inner_gt = observations["inner_gt"]
        positive_idx = inner_gt.bool()

        #################################################
        # Embeddings
        #################################################
        # Batch info
        B = observations["rgb"].shape[0]
        device = observations["rgb"].device

        # Embedding
        rgb_seq_features = self.encode_rgb(observations)
        depth_seq_features = self.encode_depth(observations)
        inst_seq_features, inst_mask = self.encode_inst(observations)
        sub_seq_features, sub_mask = self.encode_sub(observations)

        # Cached features
        self.rgb_seq_features = rgb_seq_features[positive_idx].detach().clone()
        self.depth_seq_features = depth_seq_features[positive_idx].detach().clone()
        self.inst_features = inst_seq_features[positive_idx].detach().clone()
        self.sub_features = sub_seq_features[positive_idx].detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_seq_features)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(depth_seq_features)
        instruction_embedding, feature_mask_inst = self._feature_mask(
            inst_seq_features, pad_mask=inst_mask
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_seq_features, pad_mask=sub_mask
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding_seq.shape[0], -1, -1)
        )

        #################################################
        # Vision part
        #################################################
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
        # Modality type ids
        B, L_rgb, D_rgb = rgb_embedding_seq.shape
        B, L_depth, D_depth = depth_embedding_seq.shape
        vision_type_ids = torch.cat(
            [
                torch.ones((B, L_rgb + 1), dtype=int, device=device) * RGB,
                torch.ones((B, L_depth + 1), dtype=int, device=device) * DEP,
            ],
            dim=1,
        )

        # Transformer blocks
        seq_vision_out = self._t_forward(
            seq_vision_embedding, token_type_ids=vision_type_ids
        )

        # split output sequence
        rgb_cls = seq_vision_out[:, 0, :]
        rgb_out = seq_vision_out[positive_idx, 1 : L_rgb + 1, :]
        depth_cls = seq_vision_out[:, L_rgb + 1, :]
        depth_out = seq_vision_out[positive_idx, L_rgb + 2 :, :]

        # mask feature reconstruction, only positive samples are involved
        feature_mask_rgb = feature_mask_rgb[positive_idx]
        feature_mask_depth = feature_mask_depth[positive_idx]
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb.logical_not()])
        depth_rec = self.depth_reconstruction(
            depth_out[feature_mask_depth.logical_not()]
        )
        loss_rec += F.mse_loss(
            rgb_rec, self.rgb_seq_features[feature_mask_rgb.logical_not()]
        )
        loss_rec += F.mse_loss(
            depth_rec, self.depth_seq_features[feature_mask_depth.logical_not()]
        )
        # mean feature reconstruction, only positive samples are involved
        rgb_mean_gt = self.rgb_seq_features.mean(dim=1)
        rgb_mean_rec = self.mean_rgb_reconstruction(rgb_cls[positive_idx])
        loss_mean += F.mse_loss(rgb_mean_rec, rgb_mean_gt)
        depth_mean_gt = self.depth_seq_features.mean(dim=1)
        depth_mean_rec = self.mean_depth_reconstruction(depth_cls[positive_idx])
        loss_mean += F.mse_loss(depth_mean_rec, depth_mean_gt)
        # inner alignment, all samples are involved
        rgb_depth_cls = torch.cat([rgb_cls, depth_cls], dim=1)
        inner_pre_v = self.inner_alignment_v(rgb_depth_cls)
        loss_align += F.binary_cross_entropy(
            torch.sigmoid(inner_pre_v.squeeze(1)), inner_gt.float()
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

        B, L_inst, D_inst = instruction_embedding.shape
        B, L_sub, D_sub = sub_instruction_embedding.shape
        language_type_ids = torch.cat(
            [
                torch.ones((B, L_inst + 1), dtype=int, device=device) * INS,
                torch.ones((B, L_sub + 1), dtype=int, device=device) * SUB,
            ],
            dim=1,
        )

        # Attention mask
        attn_mask = torch.ones((B, L_inst + L_sub + 2), dtype=int, device=device)
        attn_mask[:, 1 : L_inst + 1] = inst_mask
        attn_mask[:, L_inst + 2 :] = sub_mask

        # Transformer blocks
        seq_language_out = self._t_forward(
            seq_language_embedding,
            attention_mask=attn_mask,
            token_type_ids=language_type_ids,
        )
        # split output sequence
        inst_cls = seq_language_out[:, 0, :]
        inst_out = seq_language_out[positive_idx, 1 : L_inst + 1, :]
        sub_cls = seq_language_out[:, L_inst + 1, :]
        sub_out = seq_language_out[positive_idx, L_inst + 2 :, :]
        # mask feature reconstruction, only positive samples are involved
        feature_mask_inst = feature_mask_inst[positive_idx]
        feature_mask_sub = feature_mask_sub[positive_idx]
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst.logical_not()])
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub.logical_not()])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst.logical_not()])
            / COEF_REC_INST
        )
        loss_rec += F.mse_loss(
            sub_rec, self.sub_features[feature_mask_sub.logical_not()]
        )
        # mean feature reconstruction
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            inst_mask[positive_idx].sum(dim=1, keepdim=True) + EPS
        )
        inst_mean_rec = self.mean_inst_reconstruction(inst_cls[positive_idx])
        loss_mean += F.mse_loss(inst_mean_rec, inst_mean_gt)
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            sub_mask[positive_idx].sum(dim=1, keepdim=True) + EPS
        )
        sub_mean_rec = self.mean_sub_reconstruction(sub_cls[positive_idx])
        loss_mean += F.mse_loss(sub_mean_rec, sub_mean_gt)
        # inner alignment
        inst_sub_cls = torch.cat([inst_cls, sub_cls], dim=1)
        inner_pre_l = self.inner_alignment_l(inst_sub_cls)
        loss_align += F.binary_cross_entropy(
            torch.sigmoid(inner_pre_l.squeeze(1)), inner_gt.float()
        )

        return {
            "loss_rec": loss_rec if positive_idx.sum() > 0 else 0,
            "loss_mean": loss_mean if positive_idx.sum() > 0 else 0,
            "loss_align": loss_align,
        }, {
            "inner_gt_v": inner_gt,
            "inner_pre_v": inner_pre_v,
            "inner_gt_l": inner_gt,
            "inner_pre_l": inner_pre_l,
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
        B = observations["rgb"].shape[0]
        device = observations["rgb"].device

        # Embedding
        rgb_seq_features = self.encode_rgb(observations)
        depth_seq_features = self.encode_depth(observations)
        inst_seq_features, inst_mask = self.encode_inst(observations)
        sub_seq_features, sub_mask = self.encode_sub(observations)

        # Cached features
        self.rgb_seq_features = rgb_seq_features[positive_idx].detach().clone()
        self.depth_seq_features = depth_seq_features[positive_idx].detach().clone()
        self.inst_features = inst_seq_features[positive_idx].detach().clone()
        self.sub_features = sub_seq_features[positive_idx].detach().clone()

        # Masked features
        # feature masks only catch the masked positions, without padding positions
        rgb_embedding_seq, feature_mask_rgb = self._feature_mask(rgb_seq_features)
        depth_embedding_seq, feature_mask_depth = self._feature_mask(depth_seq_features)
        instruction_embedding, feature_mask_inst = self._feature_mask(
            inst_seq_features, pad_mask=inst_mask
        )
        sub_instruction_embedding, feature_mask_sub = self._feature_mask(
            sub_seq_features, pad_mask=sub_mask
        )

        # FC
        instruction_embedding = self.inst_fc(instruction_embedding)
        sub_instruction_embedding = self.sub_fc(sub_instruction_embedding)
        rgb_embedding_seq = self.rgb_fc(rgb_embedding_seq)
        depth_embedding_seq = self.depth_fc(depth_embedding_seq)

        # Token
        token_embeddings = self.token_embedding.expand(
            (rgb_embedding_seq.shape[0], -1, -1)
        )

        #################################################
        # All modalities
        #################################################
        ## Construct input sequence
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

        # Modality type ids
        B, L_rgb, D_rgb = rgb_embedding_seq.shape
        B, L_depth, D_depth = depth_embedding_seq.shape
        B, L_inst, D_inst = instruction_embedding.shape
        B, L_sub, D_sub = sub_instruction_embedding.shape

        all_type_ids = torch.cat(
            [
                torch.ones((B, L_rgb + 1), dtype=int, device=device) * RGB,
                torch.ones((B, L_depth + 1), dtype=int, device=device) * DEP,
                torch.ones((B, L_inst + 1), dtype=int, device=device) * INS,
                torch.ones((B, L_sub + 1), dtype=int, device=device) * SUB,
            ],
            dim=1,
        )
        # Attention mask
        attn_mask = torch.ones(
            (B, L_rgb + L_depth + L_inst + L_sub + 4), dtype=int, device=device
        )
        attn_mask[:, L_rgb + L_depth + 3 : L_rgb + L_depth + L_inst + 3] = inst_mask
        attn_mask[:, L_rgb + L_depth + L_inst + 4 :] = sub_mask

        # Transformer blocks
        seq_out = self._t_forward(
            seq_embedding, attention_mask=attn_mask, token_type_ids=all_type_ids
        )

        # Total feature, select
        rgb_cls = seq_out[:, 0, :]
        depth_cls = seq_out[:, L_rgb + 1, :]
        inst_cls = seq_out[:, L_rgb + L_depth + 2, :]
        sub_cls = seq_out[:, L_rgb + L_depth + L_inst + 3, :]
        rgb_out = seq_out[
            positive_idx,
            1 : L_rgb + 1,
        ]
        depth_out = seq_out[positive_idx, L_rgb + 2 : L_rgb + L_depth + 2, :]
        inst_out = seq_out[
            positive_idx, L_rgb + L_depth + 3 : L_rgb + L_depth + L_inst + 3, :
        ]
        sub_out = seq_out[positive_idx, L_rgb + L_depth + L_inst + 4 :, :]

        # mask feature reconstruction, only positive samples are involved
        feature_mask_rgb = feature_mask_rgb[positive_idx]
        feature_mask_depth = feature_mask_depth[positive_idx]
        rgb_rec = self.rgb_reconstruction(rgb_out[feature_mask_rgb.logical_not()])
        depth_rec = self.depth_reconstruction(
            depth_out[feature_mask_depth.logical_not()]
        )
        loss_rec += F.mse_loss(
            rgb_rec, self.rgb_seq_features[feature_mask_rgb.logical_not()]
        )
        loss_rec += F.mse_loss(
            depth_rec, self.depth_seq_features[feature_mask_depth.logical_not()]
        )
        feature_mask_inst = feature_mask_inst[positive_idx]
        feature_mask_sub = feature_mask_sub[positive_idx]
        inst_rec = self.inst_reconstruction(inst_out[feature_mask_inst.logical_not()])
        sub_rec = self.sub_reconstruction(sub_out[feature_mask_sub.logical_not()])
        loss_rec += (
            F.mse_loss(inst_rec, self.inst_features[feature_mask_inst.logical_not()])
            / COEF_REC_INST
        )
        loss_rec += F.mse_loss(
            sub_rec, self.sub_features[feature_mask_sub.logical_not()]
        )

        # mean feature reconstruction, only positive samples are involved
        rgb_mean_gt = self.rgb_seq_features.mean(dim=1)
        rgb_mean_rec = self.mean_rgb_reconstruction(rgb_cls[positive_idx])
        loss_mean += F.mse_loss(rgb_mean_rec, rgb_mean_gt)
        depth_mean_gt = self.depth_seq_features.mean(dim=1)
        depth_mean_rec = self.mean_depth_reconstruction(depth_cls[positive_idx])
        loss_mean += F.mse_loss(depth_mean_rec, depth_mean_gt)
        inst_mean_gt = self.inst_features.sum(dim=1) / (
            inst_mask[positive_idx].sum(dim=1, keepdim=True) + EPS
        )
        inst_mean_rec = self.mean_inst_reconstruction(inst_cls[positive_idx])
        loss_mean += F.mse_loss(inst_mean_rec, inst_mean_gt)
        sub_mean_gt = self.sub_features.sum(dim=1) / (
            sub_mask[positive_idx].sum(dim=1, keepdim=True) + EPS
        )
        sub_mean_rec = self.mean_sub_reconstruction(sub_cls[positive_idx])
        loss_mean += F.mse_loss(sub_mean_rec, sub_mean_gt)

        # inner alignment, all samples are involved
        rgb_depth_cls = torch.cat([rgb_cls, depth_cls], dim=1)
        inner_pre_v = self.inner_alignment_v(rgb_depth_cls)
        loss_inner += F.binary_cross_entropy(
            torch.sigmoid(inner_pre_v.squeeze()), inner_gt.float()
        )
        inst_sub_cls = torch.cat([inst_cls, sub_cls], dim=1)
        inner_pre_l = self.inner_alignment_l(inst_sub_cls)
        loss_inner += F.binary_cross_entropy(
            torch.sigmoid(inner_pre_l.squeeze()), inner_gt.float()
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
            "inner_pre_v": inner_pre_v,
            "inner_gt_l": inner_gt,
            "inner_pre_l": inner_pre_l,
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
