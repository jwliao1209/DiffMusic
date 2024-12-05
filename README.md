# DiffMusic: A Unified Diffusion-Based Framework for Music Inverse Problem

This repository contains the implementation for final project of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this slides.


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 diffmusic
source diffmusic/bin/activate
pip install -r requirements.txt
```


## Generating Music
To generate the music, you can run the command:
```
python run.py \
    --config_path <Path for config>
    --prompt <Text prompt>
    --negative_prompt <Negative text prompt>
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
