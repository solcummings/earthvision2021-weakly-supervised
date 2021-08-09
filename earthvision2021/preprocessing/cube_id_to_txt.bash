#!/bin/bash/

output_file="./train_with_labels.txt"

for i in */; do
    region=$(echo $i | awk -F'_' '{print $1}')
    cube=$(echo $i | awk -F'_' '{print $2"_"$3"_"$4}')
    # remove trailing slash
    cube=${cube::-1}
    echo $region"/"$cube >> $output_file
done

