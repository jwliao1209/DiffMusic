#!/bin/bash

# Download data
if [ ! -d "data" ]; then
    gdown 1DLnQKzREYul0puuTJNdBkiEogf4wBmlt -O data.zip
    unzip -n data.zip
    rm data.zip
    mv sampled_wav_files data
fi
