#!/bin/bash

models=(
    "audioldm2"
    "musicldm"
)

schedulers=(
    "dps"
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

for model in "${models[@]}"; do
    for scheduler in "${schedulers[@]}"; do
        for task in "${tasks[@]}"; do
            OUTPUT_DIR="outputs/${model}/${scheduler}/${task}"
            echo "Running evaluation: ${OUTPUT_DIR}"
            python eval.py \
                -gt "${OUTPUT_DIR}/wav_label" \
                -r "${OUTPUT_DIR}/wav_recon"
        done
    done
done
