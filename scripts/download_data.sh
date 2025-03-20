#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data

    gdown 1cFV8snb2renglGCIbGoBKgltHi-KOzfh -O moises_subset.zip
    gdown 1SSGGkh3MXaKi6u_evrhyKG8Y53OgA1_Z -O musiccaps_subset.zip

    unzip -n moises_subset.zip
    unzip -n musiccaps_subset.zip

    rm moises_subset.zip
    rm musiccaps_subset.zip

    mv moises_subset data
    mv musiccaps_subset data
fi


#if [ ! -d "data" ]; then
#    Download moises_inst_subset data
#    gdown 1ui0_9OmdiZBYIgGoyQHrw3KK-sFmvNT2 -O data.zip
#    unzip -n data.zip
#    rm data.zip
#    mv moises_inst_subset data
#fi
