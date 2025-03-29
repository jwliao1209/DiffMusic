# DiffMusic

This repository implement the diffusion-based framework for solving music inverse problems.


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
virtualenv --python=python3.10 diffmusic
source diffmusic/bin/activate
pip install -r requirements.txt
```


## Download CLAP pretrained weight
```bash
mkdir CLAP_weights
cd CLAP_weights

wget https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2022.pth

wget https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth

cd ..
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
    --task <Inverse Problem Task: {music_generation, music_inpainting, phase_retrieval, super_resolution, dereverberation, style_guidance}> \
    --config_name <Sampling Scheduler: ddim, dps, mpgd, dsg, diffmusic> \
    --prompt ""
```

### Available Inverse Problem Tasks
The following tasks can be specified with the `--task` option:
- `music_generation`
- `music_inpainting`
- `phase_retrieval`
- `super_resolution`
- `dereverberation`
- `style_guidance`

### Available Scheduler
The following tasks can be specified with the `--config_name` option:
- `ddim`
- `dps`
- `mpgd`
- `dsg`
- `diffmusic`

### Available Model Configurations
Specify the model configuration file with the `--config_path` option:
- `configs/audioldm2.yaml`
- `configs/musicldm.yaml`

### Example Command
To perform music inpainting with a specific configuration:
```bash
python run.py \
    --task "music_inpainting" \
    --config_path "configs/musicldm.yaml" \
    --prompt ""
```

To perform style guidance with a specific configuration:
```bash
python run.py \
    --task "style_guidance" \
    --config_path "configs/audioldm2.yaml" \
    --prompt "A female reporter is singing"
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{liao2024_diffmusic,
    title  = {DiffMusicIP: Exploring Conditional Diffusion Models Zero-shot Potential for Solving Music Inverse Problems},
    author = {Jia-Wei Liao, Pin-Chi Pan, and Sheng-Ping Yang},
    url    = {https://github.com/jwliao1209/DiffMusicIP},
    year   = {2024}
}
```
