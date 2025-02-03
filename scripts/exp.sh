#!/bin/bash

configs=(
    "configs/audioldm2.yaml"
#    "configs/musicldm.yaml"
)

schedulers=(
#    "dps"
#    "mpgd"
#    "dsg"
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

for config in "${configs[@]}"; do
    for scheduler in "${schedulers[@]}"; do
        for task in "${tasks[@]}"; do
            for instrument in "${instruments[@]}"; do
                echo "=================================================="
                echo "Running task: $task with config: $config"
                echo "Scheduler   : $scheduler"
                echo "Instrument  : $instrument"
                echo "Prompt      : $prompt"
                CUDA_VISIBLE_DEVICES=1 python run.py -t "$task" -c "$config" -s "$scheduler" --instrument "$instrument" --prompt "$prompt"
                echo "=================================================="
            done
        done
    done
done


echo "All tasks completed!"
