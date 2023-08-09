from copy import deepcopy
from typing import List, Optional, Union

import habitat_baselines.config.default
import numpy as np
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# ----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "data/videos/debug"
_C.DEBUG = False
_C.DEBUG_SUFFIX = "save"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.LOG_FILE = "debug/%s.log"
_C.TEST_EPISODE_COUNT = -1

# ----------------------------------------------------------------------------
# EVAL CONFIG
# ----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# ----------------------------------------------------------------------------
# INFERENCE CONFIG
# ----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "r2r"  # either 'rxr' or 'r2r'

# ----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# ----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
# number of network update rounds per iteration
_C.IL.epochs = 4
# if true, uses class-based inflection weighting
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
_C.IL.load_from_pretrain = True
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
_C.IL.max_grad_norm = 0.5
_C.IL.optimizer = "Adam"

# ----------------------------------------------------------------------------
# IL: RECOLLECT TRAINER CONFIG
# ----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = False
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)

# ----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# ----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
# dataset aggregation rounds (1 for teacher forcing)
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.start_iteration = 0
# episodes collected per iteration (size of dataset for teacher forcing)
_C.IL.DAGGER.update_size = 5000
# probability of taking the expert action (1.0 for teacher forcing)
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.lmdb_map_size = 1.2e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens.
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = "data/trajectories_dirs/debug/trajectories.lmdb"
_C.IL.DAGGER.drop_existing_lmdb_features = True
# ----------------------------------------------------------------------------
# RL CONFIG
# ----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "waypoint_reward_measure"
_C.RL.SUCCESS_MEASURE = "success"
_C.RL.NUM_UPDATES = 200000
_C.RL.LOG_INTERVAL = 10
_C.RL.CHECKPOINT_INTERVAL = 250
_C.RL.SUCCESS_REWARD = 2.5
_C.RL.SLACK_REWARD = -0.01
_C.RL.MEASURE_RATIO: 1.0
# -----------------------------------------------------------------------------
# preemption CONFIG
# -----------------------------------------------------------------------------
# _C.RL.preemption = CN()
# # Append the slurm job ID to the resume state filename if running a slurm job
# # This is useful when you want to have things from a different job but same
# # same checkpoint dir not resume.
# _C.RL.preemption.append_slurm_job_id = False
# # Number of gradient updates between saving the resume state
# _C.RL.preemption.save_resume_state_interval = 100
# # Save resume states only when running with slurm
# # This is nice if you don't want debug jobs to resume
# _C.RL.preemption.save_state_batch_only = False
# # ----------------------------------------------------------------------------
# POLICY CONFIG
# ----------------------------------------------------------------------------
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = []
_C.RL.POLICY.OBS_TRANSFORMS.OBS_STACK = CN()
_C.RL.POLICY.OBS_TRANSFORMS.OBS_STACK.SENSOR_REWRITES = [
    (
        "rgb",
        [
            "rgb",
            "rgb_1",
            "rgb_2",
            "rgb_3",
            "rgb_4",
            "rgb_5",
            "rgb_6",
            "rgb_7",
            "rgb_8",
            "rgb_9",
            "rgb_10",
            "rgb_11",
        ],
    ),
    (
        "depth",
        [
            "depth",
            "depth_1",
            "depth_2",
            "depth_3",
            "depth_4",
            "depth_5",
            "depth_6",
            "depth_7",
            "depth_8",
            "depth_9",
            "depth_10",
            "depth_11",
        ],
    ),
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
# ----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# ----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 2
_C.RL.PPO.num_mini_batch = 4
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.clip_value_loss = True
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.pano_entropy_coef = 1.0
_C.RL.PPO.offset_entropy_coef = 0.0
_C.RL.PPO.distance_entropy_coef = 0.0
_C.RL.PPO.lr = 2.0e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.2
_C.RL.PPO.num_steps = 16
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = False
# regularize offset. maximum loss: 0.2618 * 0.1146 = 0.03
_C.RL.PPO.offset_regularize_coef = 0.1146
# ----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# ----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "NCCL"  # or GLOO
_C.RL.DDPPO.reset_critic = True
_C.RL.DDPPO.start_from_requeue = False
_C.RL.DDPPO.requeue_path = "data/interrupted_state.pth"

# ----------------------------------------------------------------------------
# PRETRAIN CONFIG
# ----------------------------------------------------------------------------

_C.PRETRAIN = CN()
_C.PRETRAIN.is_requeue = False
_C.PRETRAIN.stage = "NONE"
_C.PRETRAIN.excludes = [
    "clip_encoder",
    "depth_encoder",
    "rgb_reconstruction",
    "depth_reconstruction",
    "inst_reconstruction",
    "sub_reconstruction",
    "mean_rgb_reconstruction",
    "mean_depth_reconstruction",
    "mean_inst_reconstruction",
    "mean_sub_reconstruction",
    "inner_alignment",
    "outer_alignment",
]
_C.PRETRAIN.masked_feature_ratio = 0.15
_C.PRETRAIN.max_grad_norm = 1.0  # cat or add

_C.PRETRAIN.STAGE0 = CN()
_C.PRETRAIN.STAGE0.load_from_ckpt = False
_C.PRETRAIN.STAGE0.ckpt_to_load = ""
_C.PRETRAIN.STAGE0.epochs = 5
_C.PRETRAIN.STAGE0.lr = 1e-4
_C.PRETRAIN.STAGE0.warmup = 1000
_C.PRETRAIN.STAGE0.batch_size = 32
_C.PRETRAIN.STAGE0.loss_weights = [1.0, 1.0]
_C.PRETRAIN.STAGE0.train_frac = 1.0
_C.PRETRAIN.STAGE0.folder = (
    "/root/autodl-tmp/stage0"  # must contains rgb.mat, depth.mat, inst.mat, sub.mat,
)

_C.PRETRAIN.STAGE1 = CN()
_C.PRETRAIN.STAGE1.load_from_ckpt = False
_C.PRETRAIN.STAGE1.ckpt_to_load = ""
_C.PRETRAIN.STAGE1.epochs = 5
_C.PRETRAIN.STAGE1.lr = 1e-4
_C.PRETRAIN.STAGE1.warmup = 1000
_C.PRETRAIN.STAGE1.batch_size = 32
_C.PRETRAIN.STAGE1.loss_weights = [0.5, 0.5, 1.0]
_C.PRETRAIN.STAGE1.train_frac = 1.0
_C.PRETRAIN.STAGE1.folder = (
    "/root/autodl-tmp/stage1"  # must contains rgb_depth_large.mat, inst_sub_large.mat
)
_C.PRETRAIN.STAGE1.positive_ratio = 0.4

_C.PRETRAIN.STAGE2 = CN()
_C.PRETRAIN.STAGE2.load_from_ckpt = False
_C.PRETRAIN.STAGE2.ckpt_to_load = ""
_C.PRETRAIN.STAGE2.epochs = 5
_C.PRETRAIN.STAGE2.lr = 1e-4
_C.PRETRAIN.STAGE2.warmup = 1000
_C.PRETRAIN.STAGE2.batch_size = 32
_C.PRETRAIN.STAGE2.loss_weights = [0.4, 0.4, 1.0, 1.0]
_C.PRETRAIN.STAGE2.train_frac = 1.0
_C.PRETRAIN.STAGE2.folder = "/root/autodl-tmp/stage2"  # must contains data.mat
_C.PRETRAIN.STAGE2.positive_ratio = 0.3
_C.PRETRAIN.STAGE2.inner_ratio = 0.5

# ----------------------------------------------------------------------------
# MODELING CONFIG
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"
_C.MODEL.normalize_rgb = False

_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False
_C.MODEL.ablate_sub_instruction = False


_C.MODEL.ablate_mla = False
_C.MODEL.ablate_ssa = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_FSASub/embeddings200d.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_FSASub/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 200
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128  # in type0, it is 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = True

_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256
_C.MODEL.RGB_ENCODER.trainable = False

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 2048
_C.MODEL.DEPTH_ENCODER.feature_size = 256
_C.MODEL.DEPTH_ENCODER.single_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = "data/ddppo-models/gibson-4plus-resnet50.pth"
_C.MODEL.DEPTH_ENCODER.trainable = False
_C.MODEL.DEPTH_ENCODER.final_relu = True

_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.model_name = "ViT-B/32"
_C.MODEL.CLIP.output_size = 512
_C.MODEL.CLIP.feature_size = 256  # in type0, it is 256
_C.MODEL.CLIP.vit_size = 768
_C.MODEL.CLIP.trainable = False
_C.MODEL.CLIP.downsample_size = 3  # 3x3 = 9
_C.MODEL.CLIP.rgb_level = -1  # final level feature


_C.MODEL.EVOENC = CN()
_C.MODEL.EVOENC.window_size = 8
_C.MODEL.EVOENC.instruction_len = 77
_C.MODEL.EVOENC.sub_len = 12
_C.MODEL.EVOENC.hidden_size = 512
_C.MODEL.EVOENC.layers = 3
_C.MODEL.EVOENC.heads = 8
_C.MODEL.EVOENC.learnable_pe = True
_C.MODEL.EVOENC.learnable_mask = False
_C.MODEL.EVOENC.pe_type = "pt"  # position, token, split_position, pt, none
_C.MODEL.EVOENC.rgb_len = 10
_C.MODEL.EVOENC.depth_len = 16
_C.MODEL.EVOENC.dropout = 0.0
_C.MODEL.EVOENC.inner_dropout = 0.0
_C.MODEL.EVOENC.pre_ln = True
_C.MODEL.EVOENC.pre_dropout = 0.1
_C.MODEL.EVOENC.post_ln = True
_C.MODEL.EVOENC.post_dropout = 0.0
_C.MODEL.EVOENC.aggregate = "cat"  # cat or add
_C.MODEL.EVOENC.freeze_weights = -1  # -1: not freeze;
# 0: freeze layers before transformer
# 1: freeze preLN
# x>=1: freeze blcks[0:x]
_C.MODEL.EVOENC.prev_action = True
_C.MODEL.EVOENC.post_fusion = True

_C.MODEL.BERT = CN()
_C.MODEL.BERT.model_name = "bert-base-uncased"
_C.MODEL.BERT.output_size = 768
_C.MODEL.BERT.use_layer = 6
_C.MODEL.BERT.tune_layer = []
_C.MODEL.BERT.use_fc = True
_C.MODEL.BERT.use_cls = False

# Different states are stacked at layer dim, so their feature sizes must be same.
_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type_low = "GRU"
_C.MODEL.STATE_ENCODER.num_layers_low = 1
_C.MODEL.STATE_ENCODER.rnn_type_high = "GRU"
_C.MODEL.STATE_ENCODER.num_layers_high = 1
_C.MODEL.STATE_ENCODER.rnn_type_action = "GRU"
_C.MODEL.STATE_ENCODER.num_layers_action = 1
_C.MODEL.STATE_ENCODER.dropout_ratio = 0.1

_C.MODEL.MLA = CN()
_C.MODEL.MLA.heads = 8
_C.MODEL.MLA.feature_drop = 0.0
_C.MODEL.MLA.hidden_size = 512

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier \theta
_C.MODEL.PEAK_ATTENTION = CN()
_C.MODEL.PEAK_ATTENTION.use = False
_C.MODEL.PEAK_ATTENTION.gt = True
_C.MODEL.PEAK_ATTENTION.curve = (
    "gaussian"  # gaussian, constant, linear, quadratic, cubic
)
_C.MODEL.PEAK_ATTENTION.alpha = 1.0  # loss multiplier \lambda
_C.MODEL.PEAK_ATTENTION.sigma = 0.6  # sigma
_C.MODEL.PEAK_ATTENTION.type = (
    0  # the order of polynomial; 0,1,2,3 mean constant, linear, quadratic or cubic
)
_C.MODEL.PEAK_ATTENTION.steps = 60840
_C.MODEL.PEAK_ATTENTION.threshold = 0.25

_C.MODEL.FOCAL = CN()
_C.MODEL.FOCAL.gamma = 0.0
_C.MODEL.FOCAL.alpha = None

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False
_C.MODEL.SEQ2SEQ.encoder_prev_action = False
_C.MODEL.SEQ2SEQ.decoder_prev_action = False

_C.MODEL.WAYPOINT = CN()
# if False, the distance is 0.25m (heading prediction network, HPN)
_C.MODEL.WAYPOINT.predict_distance = True
# if True, predict distance from a truncated normal distribution.
# if False, predict from discrete categorical candidates.
_C.MODEL.WAYPOINT.continuous_distance = True
_C.MODEL.WAYPOINT.min_distance_var = 0.0625  # a stddev of 0.25m
_C.MODEL.WAYPOINT.max_distance_var = 3.52  # a stddev of (range / 2)
_C.MODEL.WAYPOINT.max_distance_prediction = 2.75
_C.MODEL.WAYPOINT.min_distance_prediction = 0.25
# 6 distances gives 0.5m increments for a distance range of [0.25, 2.75]
_C.MODEL.WAYPOINT.discrete_distances = 6
# if False, predict heading from 12 equiangular candidates.
_C.MODEL.WAYPOINT.predict_offset = True
_C.MODEL.WAYPOINT.continuous_offset = True
_C.MODEL.WAYPOINT.min_offset_var = 0.0110  # stddev of 6 degrees
_C.MODEL.WAYPOINT.max_offset_var = 0.0685  # stddev of (range / 2)
# 7 offsets gives 5deg increments for an offset range [-15deg, 15deg]
_C.MODEL.WAYPOINT.discrete_offsets = 7
_C.MODEL.WAYPOINT.offset_temperature = 1.0


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config


def add_pano_sensors_to_config(config: CN) -> CN:
    """Dynamically adds RGB and Depth cameras to config.TASK_CONFIG, forming
    an N-frame panorama. The PanoRGB and PanoDepth observation transformers
    can be used to stack these frames together.
    """
    num_cameras = config.TASK_CONFIG.TASK.PANO_ROTATIONS

    config.defrost()
    orient = [(0, np.pi * 2 / num_cameras * i, 0) for i in range(num_cameras)]
    sensor_uuids = ["rgb"]
    if "RGB_SENSOR" in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, num_cameras):
            camera_template = f"RGB_{camera_id}"
            camera_config = deepcopy(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]

            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)
            setattr(config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
            config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

    sensor_uuids = ["depth"]
    if "DEPTH_SENSOR" in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, num_cameras):
            camera_template = f"DEPTH_{camera_id}"
            camera_config = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]
            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)

            setattr(config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
            config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

    config.SENSORS = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    config.freeze()
    return config
