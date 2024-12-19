# DiffMusic: A Zero-shot Diffusion-Based Framework for Music Inverse Problem

This repository contains the implementation for final project of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides](https://docs.google.com/presentation/d/1djOZFM2DMYFEPwKalCmMgZuUplhSQCBZlv7DMsROLuQ/edit?usp=sharing).


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
virtualenv --python=python3.10 diffmusic
source diffmusic/bin/activate
pip install -r requirements.txt
```


## Data Preparation
To download the dataset, run the following script:
```
bash scripts/download_data.sh
```


## Generating Music for Inverse Problems

To address an inverse problem, you can use the following command:

```bash
python run.py \
    --task <Inverse Problem Task: {music_inpainting, phase_retrieval, super_resolution, dereverberation, source_separation}> \
    --scheduler <Sampling Scheduler: dps, mpgd> \
    --config_path <Path to Model Configuration> \
    --prompt ""
```

### Available Inverse Problem Tasks
The following tasks can be specified with the `--task` option:
- `music_inpainting`
- `phase_retrieval`
- `super_resolution`
- `dereverberation`
- `source_separation`

### Available Model Configurations
Specify the model configuration file with the `--config_path` option:
- `configs/audioldm2.yaml`
- `configs/musicldm.yaml`

### Example Command
To perform music inpainting with a specific configuration:
```bash
python run.py \
    --task "music_inpainting" \
    --config_path "configs/audioldm2.yaml" \
    --prompt ""
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{liao2024_diffmusic,
    title  = {DiffMusic: A Unified Diffusion-Based Framework for Music Inverse Problem},
    author = {Jia-Wei Liao, Pin-Chi Pan, and Sheng-Ping Yang},
    url    = {https://github.com/jwliao1209/DiffMusic},
    year   = {2024}
}
```
