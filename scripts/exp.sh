#!/bin/bash

config_names=(
    "dps"
    "mpgd"
    "dsg"
    "ditto"
    "diffmusic"
)

tasks=(
    "music_inpainting"
    "super_resolution"
    "phase_retrieval"
    "music_dereverberation"
)

show_progress=True
prompt=""

for config_name in "${config_names[@]}"; do
    for task in "${tasks[@]}"; do
        if [ "$show_progress" = "True" ]; then
            CUDA_VISIBLE_DEVICES=8 python run.py -t "$task" -c "$config_name" --prompt "$prompt" --show_progress
        else
            CUDA_VISIBLE_DEVICES=8 python run.py -t "$task" -c "$config_name" --prompt "$prompt"
        fi
    done
done

echo "All tasks completed!"

