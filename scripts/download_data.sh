#!/bin/bash

# Download data
if [ ! -d "data" ]; then
    gdown 1rBJxgnUpvsrNePEEPLBzRCEqSzLsSp1m -O data.zip
    unzip -n data.zip
    rm data.zip
    mv moises_inst_subset data
fi
