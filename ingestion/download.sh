#!/bin/bash

## Download data
echo "Downloading data..."
tmux new-session -s dl -n bash -d
tmux send-keys -t dl:0 'wget -O-  "https://www.cs.helsinki.fi/group/carat/data-sharing/carat-data-top1k-users-2014-to-2018-08-25.zip" | aws s3 cp - s3://hxzhuang/carat/carat.zip'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz" | aws s3 cp - s3://hxzhuang/rico/unique_uis.tar.gz'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv" | aws s3 cp - s3://hxzhuang/rico/ui_details.csv'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip" | aws s3 cp - s3://hxzhuang/rico/ui_layout_vectors.zip'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/traces.tar.gz" | aws s3 cp - s3://hxzhuang/rico/traces.tar.gz'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/animations.tar.gz" | aws s3 cp - s3://hxzhuang/rico/animations.tar.gz'
tmux send-keys -t dl:0 'wget -O-  "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv" | aws s3 cp - s3://hxzhuang/rico/app_details.csv'
tmux send-keys -t dl:0 'wget -O-  "https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip" | aws s3 cp - s3://hxzhuang/rico/semantic_annotations.zip'

## error handling
if [ $? == 0 ]; then
    echo "Download success!"
else
    echo "Error in downloading!"
fi