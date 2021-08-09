#!/bin/bash


in_dir='../earthvision2021/data/source/'
out_dir='../earthvision2021/data/train/'
download_address='https://competitions.codalab.org/competitions/30441#participate'

if [ -d $in_dir ]; then
    if [ -d $in_dir'/Labels/' ] && [ -d $in_dir'/planet/' ] &&
        [ -d $in_dir'/planet_test/' ]; then
        echo "Data directory $in_dir exists."
        download=false
        # convert training tif to .npy files for training phase
        python tif_to_np.py --data_dir $in_dir'/planet/' \
            --out_basepath $out_dir'/images/'
        # remove csv
        rm ./mean_std.csv
        # convert test tif
        python tif_to_np.py --data_dir $in_dir'/planet_test/' \
            --out_basepath $out_dir'/images/'
        # remove csv
        rm ./mean_std.csv
        # convert label png
        python png_to_np.py --data_dir $in_dir'/Labels/' \
            --out_basepath $out_dir'/labels/'
    else
        echo "Data directory $in_dir exists, but data is missing."
        echo "Please download the data below into the data directory and rerun $0"
        echo "Data can be found at: $download_address"
        download=true
    fi
else
    echo "Data directory $in_dir does not exist."
    echo "Please download the data below into the data directory and rerun $0"
    echo "Data can be found at: $download_address"
    mkdir --parents $in_dir
    download=true
fi

