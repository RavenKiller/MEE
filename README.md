<div align="center">

<h1>Multimodal Evolutionary Encoder for Continuous Vision-Language Navigation</h1>

<div>
    <a href='https://ieeexplore.ieee.org/document/10802484' target='_blank'>[Paper (IROS2024)]</a>
    <a href='https://ravenkiller.github.io/MEE-Project/' target='_blank'>[Project page]</a>
</div>
</div>

<br />
<div align="center">
    <img src="https://github.com/user-attachments/assets/2bf671d1-5a01-4c00-9537-efb69410c15d", width="1000", alt="TAC pre-training">
</div>
<br />

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
git clone https://github.com/RavenKiller/MEE.git
cd MEE
pip install -r requirements.txt
```
5. Download Matterport3D scenes:
   + Get the official `download_mp.py` from [Matterport3D project webpage](https://niessner.github.io/Matterport/)
   + Download scene data for Habitat
    ```bash
    # requires running with python 2.7
    python download_mp.py --task habitat -o data/scene_datasets/mp3d/
    ```
   + Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.
6. Download pre-processed episodes from [here](https://www.jianguoyun.com/p/DRKVWtQQhY--CRiE0voEIAA). Extract it into `data/datasets/`.
7. Download the depth encoder from [here](https://www.jianguoyun.com/p/DREiSbAQhY--CRjv0foEIAA). Extract the model to `data/ddppo-models/gibson-4plus-resnet50.pth`.

## Evolutionary pre-training dataset
We proposed an evolutionary pre-training strategy in this work and developed the corresponding datasets. The data collecting scripts are stored in `scripts/` with filenames like `evo_data_stage1.ipynb`.

### V1
The [v1 version](https://pan.baidu.com/s/1smZFxuhxsPaF6dSjI0QHUw) (default access code: `evop`) contains a total of 4.8M samples of all modalities. All data is organized in HDF5 format. The total size after decompression is around 720 GB. Below is the file list:
+ stage1.zip
    + rgb.mat: contains RGB data with shape (395439, 224, 224, 3)
    + depth.mat: contains depth data with shape (417900, 256, 256, 1)
    + inst.mat: contains instruction data with shape (400250, 77), zero-padded, and tokenized
    + sub.mat: contains sub-instruction data with shape (410357, 12, 77)
+ stage2.zip
    + rgb_depth_large.mat: contains aligned RGB and depth data, a total of 230766 pairs
    + inst_sub_large.mat: contains aligned instruction and sub-instruction data, a total of 157877 pairs
    + rgb_depth.mat: contains a small debug version
    + inst_sub.mat: contains a small debug version
+ stage3.zip
    + data.mat: contains aligned (RGB, depth, instruction, sub-instruction), a total of 601038 tuples 

The data source includes:
+ stage 1: [COCO](https://cocodataset.org/#home), [VisualGenome](https://visualgenome.org/), [RGBD1K](https://github.com/xuefeng-zhu5/RGBD1K), [SceneNet Depth](https://robotvault.bitbucket.io/scenenet-rgbd.html), and [BookCorpus](https://huggingface.co/datasets/bookcorpus).
+ stage 2: [NYUv2](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat), [DIODE](https://diode-dataset.org/), [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset/download), [Bonn RGB-D Dynamic](http://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/), [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html),[Touchdown](https://github.com/lil-lab/touchdown), [map2seq](https://map2seq.schumann.pub/dataset/download/), [CHALET](https://github.com/lil-lab/chalet), [Talk the Walk](https://github.com/facebookresearch/talkthewalk), and [ALFRED](https://github.com/askforalfred/alfred).
+ stage 3: [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [EnvDrop](https://github.com/airsplay/R2R-EnvDrop).

### V2
The [v2 version](https://pan.baidu.com/s/14RmyVNhOjpKJz2IFqU_gQg) contains a total of 83.9M samples of all modalities, which is a superset of v1.
All data are stored in seperated files (RGB: JPEG, Depth: PNG, Instruction: TXT, Sub-instruction: TXT). 
Collecting and loading scripts are developed in the dev branch.

Additional data sources:
[ImageNet](https://www.image-net.org/), [LAION-HighResolution](https://huggingface.co/datasets/laion/laion-high-resolution), [CC-12M](https://github.com/google-research-datasets/conceptual-12m), [C4](https://www.tensorflow.org/datasets/catalog/c4), [HM3D](https://aihabitat.org/datasets/hm3d/), [SUN3D](https://sun3d.cs.princeton.edu/), [ScanNet](http://www.scan-net.org/), [Marky-gibson](https://github.com/google-research-datasets/RxR/blob/main/marky-mT5/README.md).

Access of several datasets are subject to specific terms and conditions (e.g., HM3D). Please request the access before using them.

## Train, evaluate and test
`run.py` is the program entrance. You can run it like this:
```bash
python run.py \
  --exp-config {config} \
  --run-type {type}
```
`{config}` should be replaced by a config file path; `{type}` should be `train`, `eval`, or `inference`, meaning train, evaluate, and test models.

Our config files are stored in `evoenc/config/`:
| File | Meaning |
| ---- | ---- |
| `evoenc.yaml` | Training model with behavior cloning |
| `evoenc_da.yaml` | Training model with DAgger |
| `evoenc_aug.yaml` | Training model with EnvDrop |
| `evoenc_p{x}.yaml` | Evolutionary pre-training stage {x}+1 |
| `evoenc_p{x}_tune.yaml` | Task fine-tuning with DAgger |

Several paths (like pre-training data folder and checkpoint paths) are configured by the above YAML files or the `evoenc/config/default.py`. Remember to change them as needed.

## Pre-trained weights
\[[stage 1](https://www.jianguoyun.com/p/DQDYoIIQhY--CRiy0_oEIAA)\] \[[stage 2](https://www.jianguoyun.com/p/DYfUQDQQhY--CRi80_oEIAA)\] \[[stage 3](https://www.jianguoyun.com/p/DfU_ZLgQhY--CRjB0_oEIAA)\]

We release pre-trained encoder weights after evolutionary pre-training. We exclude the frozen pre-extractor in these weights to reduce the storage cost. Refer to the code `evoenc/models/evoenc_policy.py` to load pre-trained weights.

## Visualization
### Unified feature spaces
<img src="https://github.com/RavenKiller/MEE/assets/41775391/d26de2a4-d687-45eb-b2e9-3b22b68a929d" alt="unified" width="600">


### Evolved encoder performance
<img src="https://github.com/RavenKiller/MEE/assets/41775391/1364edc2-c561-4e09-9241-f0d306be8652" alt="unified" width="600">




### Comparison with the baseline:


https://github.com/RavenKiller/MEE/assets/41775391/a82a43fb-a2cc-45b4-80c1-d2181383e9ce





### Failure cases

Premature stop


https://github.com/RavenKiller/MEE/assets/41775391/d44df089-c92e-4094-aee6-259041afb9f4



Wrong exploration


https://github.com/RavenKiller/MEE/assets/41775391/a067abd2-73d4-4e10-a39f-8bc8aaddb0a5



Deadlock


https://github.com/RavenKiller/MEE/assets/41775391/8cbd4441-720e-4e91-9095-291b28628517





## Real scene navigation

### Alkaid Robot
<img src="https://github.com/RavenKiller/MEE/assets/41775391/8cc26483-1e67-49b7-8ad6-8f137809d3a3" alt="unified" width="400">


Alkaid is a self-developed interactive service robot. Here are some parameters:

+ Camera: 720P resolution, 90° max FOV
+ Screen: 1080P, touch screen
+ Microphone: 4-microphone circular array, 61dB SNR
+ Speaker: 2 stereo units, 150Hz-20kHz output
+ Chassis: 2-wheel differential drive, 0.5m/s max speed, 1.2rad/s max angular speed

### Demonstration

Currently, we release [13 paths](https://www.jianguoyun.com/p/DcB0_TwQlY_kBxivhrsFIAA) with VLN-CE format. The video below demonstrates 4 paths.



https://github.com/RavenKiller/MEE/assets/41775391/cda47ac0-bd28-48a3-b498-a3990ec81f61

## Citation
```
@INPROCEEDINGS{10802484,
  author={He, Zongtao and Wang, Liuyi and Chen, Lu and Li, Shu and Yan, Qingqing and Liu, Chengju and Chen, Qijun},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Multimodal Evolutionary Encoder for Continuous Vision-Language Navigation}, 
  year={2024},
  volume={},
  number={},
  pages={1443-1450},
  keywords={Visualization;Costs;Codes;Navigation;Service robots;Linguistics;Feature extraction;Solids;Decoding;Intelligent robots},
  doi={10.1109/IROS58592.2024.10802484}}

```

