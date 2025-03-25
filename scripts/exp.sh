#!/bin/bash

config_names=(
#    "dps"
#    "mpgd"
#    "dsg"
    "ditto"
#    "diffmusic"
)

tasks=(
    "music_inpainting"
    "super_resolution"
    "phase_retrieval"
    "music_dereverberation"
)

show_progress=True
supervised_space="mel_spectrogram"  # mel_spectrogram, wav_form
prompt_type="null_text"  # null_text, tag, clap
prompt=""
gpu_id=1
datasets="moises"  # moises, music_data
model="musicldm"  # audioldm2, musicldm

for config_name in "${config_names[@]}"; do
    for task in "${tasks[@]}"; do
        if [ "$show_progress" = "True" ]; then
            CUDA_VISIBLE_DEVICES="$gpu_id" python run.py -t "$task" -c "$config_name" \
            -d "$datasets" -m "$model" --supervised_space "$supervised_space" --prompt_type "$prompt_type" --prompt "$prompt" --show_progress
        else
            CUDA_VISIBLE_DEVICES="$gpu_id" nohup python -u run.py -t "$task" -c "$config_name" \
            -d "$datasets" -m "$model" --supervised_space "$supervised_space" --prompt_type "$prompt_type" --prompt "$prompt" > outputs_"$gpu_id"_"$config_name"_tag.txt &
        fi
    done
done

echo "All tasks completed!"

