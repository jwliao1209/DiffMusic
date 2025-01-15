#!/bin/bash

configs=(
    "configs/audioldm2.yaml"
    "configs/musicldm.yaml"
)

schedulers=(
#    "dps"
    "mpgd"
#    "dsg"
#    "diffmusic"
)

tasks=(
    "music_inpainting"
    "super_resolution"
    "phase_retrieval"
    "music_dereverberation"
)

instruments=(
    "bass"
    "bowed_strings"
    "drums"
    "guitar"
    "percussion"
    "piano"
    "wind"
)

for config in "${configs[@]}"; do
    for scheduler in "${schedulers[@]}"; do
        for task in "${tasks[@]}"; do
            for instrument in "${instruments[@]}"; do
                echo "=================================================="
                echo "Running task: $task with config: $config"
                echo "Scheduler   : $scheduler"
                echo "Instrument  : $instrument"
                echo "Prompt      : music of a $instrument"
                CUDA_VISIBLE_DEVICES=0 python run.py -t "$task" -c "$config" -s "$scheduler" --instrument "$instrument" --prompt "music of a $instrument" --sigma 0.0001
                echo "=================================================="
            done
        done
    done
done


echo "All tasks completed!"
