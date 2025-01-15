#!/bin/bash

# Download data
if [ ! -d "data" ]; then
    gdown 1ui0_9OmdiZBYIgGoyQHrw3KK-sFmvNT2 -O data.zip
    unzip -n data.zip
    rm data.zip
    mv moises_inst_subset data
fi
