# Unified Multi-modal Encoder by Evolutionary Pre-training for Continuous Vision-Language Navigation


## Setup
1. Use [anaconda](https://anaconda.org/) to create a Python 3.8 environment:
```bash
conda create -n habitat python3.8
conda activate habitat
```
2. Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.1) 0.2.1:
```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.2.1 headless
```
3. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.1) 0.2.1:
```bash
git clone --branch v0.2.1 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```
4. Clone this repository and install python requirements:
```bash
git clone https://github.com/RavenKiller/EvoEnc.git
cd EvoEnc
pip install -r requirements.txt
```
5. Download Matterport3D sences:
   + Get the official `download_mp.py` from [Matterport3D project webpage](https://niessner.github.io/Matterport/)
   + Download scene data for Habitat
    ```bash
    # requires running with python 2.7
    python download_mp.py --task habitat -o data/scene_datasets/mp3d/
    ```
   + Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.
6. Download preprocessed episodes from [here](https://www.jianguoyun.com/p/DRKVWtQQhY--CRiE0voEIAA). Extract it into `data/datasets/`.
7. Download the depth encoder from [here](https://www.jianguoyun.com/p/DREiSbAQhY--CRjv0foEIAA). Extract the model to `data/ddppo-models/gibson-4plus-resnet50.pth`.

## Evo Dataset
In this work, we proposed an evolutionary pre-training strategy and develop the corresponding datasets. The data collecting scripts are stored in `scripts/`, with filenames like `evo_data_stage0.ipynb`. Stage0 here corresponds to stage 1 in the paper.

Our pre-processed version contains totally 4.8M samples of all modalities. They can be download from [BaiduNetdisk](https://pan.baidu.com/s/1qvbu_z0M_2bWecNGVWKvFQ) with extraction code `mg27`. All data is organized by HDF5. The total size after decompression is around GB. Below are file list:
+ stage0.zip
    + rgb.mat: contains RGB data with shape (395439, 224, 224, 3)
    + depth.mat: contains depth data with shape (417900, 256, 256, 1)
    + inst.mat: contains instruction data with shape (400250, 77), zero-padded and tokenized
    + sub_inst.mat: contains sub-instruction data with shape (410357, 12, 77)
+ stage1.zip
    + rgb_depth_large.mat: contains aligned RGB and depth data, totally 230766 pairs
    + inst_sub_large.mat: contains aligned instruction and sub-instruction data, totally 157877 pairs
    + rgb_depth.mat: contains a small debug version
    + inst_sub.mat: contains a small debug version
+ stage2.zip
    + data.mat: contains aligned (RGB, depth, instruction, sub-instruction), totally 601038 tuples 

The data source includes  
+ stage 1: COCO [29], VisualGenome
[30], RGBD1K [31], SceneNet Depth [32] and BookCorpus
[33], 
+ stage 2:  NYUv2 [35],
DIODE [36], TUM RGB-D [37], Bonn RGB-D Dynamic
[38], SceneNet RGB-D [32], Touchdown [20], map2seq [39],
CHALET [40], Talk the Walk [41], and ALFRED [42]
+ stage 3:  VLN-CE [2] and EnvDrop [43]

## Train, evaluate and test
`run.py` is the program entrance. You can run it like:
```bash
python run.py \
  --exp-config {config} \
  --run-type {type}
```
`{config}` should be replaced by a config file path; `{type}` should be `train`, `eval` or `inference`, meaning train models, evaluate models and test models.

Our config files are stored in `evoenc/config/`:
| File | Meaning |
| ---- | ---- |
| `evoenc.yaml` | Train model with behavior cloning |
| `evoenc_da.yaml` | Train model with vanilla DAgger |
| `evoenc_aug.yaml` | Collect trajectories by EnvDrop augmentation |
| `evoenc_p0.yaml` | Evolutionary pre-training stage 1 |
| `evoenc_p1.yaml` | Evolutionary pre-training stage 2 |
| `evoenc_p2.yaml` | Evolutionary pre-training stage 3 |
| `evoenc_p{x}_tune.yaml` | Fine-tune model with  vanilla DAgger |

Several paths (like pre-training data folder, checkpoint paths) are configured by above YAML files or the `evoenc/config/default.py`. Remenber to change them as needed.

## Pre-trained weights
\[[stage 1](https://www.jianguoyun.com/p/DQDYoIIQhY--CRiy0_oEIAA)\] \[[stage 2](https://www.jianguoyun.com/p/DYfUQDQQhY--CRi80_oEIAA)\] \[[stage 2](https://www.jianguoyun.com/p/DfU_ZLgQhY--CRjB0_oEIAA)\] <br/> Access code: `evoenc`

We release pre-trained Enc weights after Evo. To reduce the storage cost, we exclude frozon pre-extractor in these weights. Refer to the code `evoenc/models/evoenc_policy.py` to load pre-trained weights.