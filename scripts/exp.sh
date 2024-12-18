#!/bin/bash

configs=(
    "configs/audioldm2.yaml"
    "configs/musicldm.yaml"
)

schedulers=(
    "dps"
    "mpgd"
    "dsg"
)

tasks=(
    "music_inpainting"
    "super_resolution"
    "phase_retrieval"
    "source_separation"
    "music_dereverberation"
)

for config in "${configs[@]}"; do
    for scheduler in "${schedulers[@]}"; do
        for task in "${tasks[@]}"; do
            echo "=================================================="
            echo "Running task: $task with config: $config"
            python run.py -t "$task" -c "$config" -s "$scheduler"
            echo "=================================================="
        done
    done
done

echo "All tasks completed!"
