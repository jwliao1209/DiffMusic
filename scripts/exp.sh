#!/bin/bash

config_names=(
    # "dps"
    # "mpgd"
    # "dsg"
    "diffmusic"
)

tasks=(
    "music_inpainting"
#    "super_resolution"
#    "phase_retrieval"
#    "music_dereverberation"
)

instruments=(
    "bass"
#    "bowed_strings"
#    "drums"
#    "guitar"
#    "percussion"
#    "piano"
#    "wind"
)

prompt=""  # "music of a $instrument"

for config_name in "${config_names[@]}"; do
    for task in "${tasks[@]}"; do
        for instrument in "${instruments[@]}"; do
            echo "=================================================="
            echo "Running task: $task with config: $config"
            echo "Scheduler   : $config_name"
            echo "Instrument  : $instrument"
            echo "Prompt      : $prompt"
            CUDA_VISIBLE_DEVICES=0 python run.py -t "$task" -c "$config_name" --instrument "$instrument" --prompt "$prompt"
            echo "=================================================="
        done
    done
done

echo "All tasks completed!"
