#!/bin/bash

models=(
    "audioldm2"
#    "musicldm"
)

datasets=(
    "moises"
#    "musiccaps"
)

config_names=(
    "dps"
#    "mpgd"
#    "dsg"
    "ditto"
    "diffmusic"
)

tasks=(
    "music_inpainting"
#    "super_resolution"
#    "phase_retrieval"
#    "music_dereverberation"
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for config_name in "${config_names[@]}"; do
            for task in "${tasks[@]}"; do
                OUTPUT_DIR="outputs_ablation_wav_form/${model}/${dataset}/${config_name}/${task}"
                echo "Running evaluation: ${OUTPUT_DIR}"
                python eval.py \
                    -gt "${OUTPUT_DIR}/wav_label" \
                    -r "${OUTPUT_DIR}/wav_recon"
            done
        done
    done
done
