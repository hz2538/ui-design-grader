#!/bin/bash

## Download data, make sure you have enough space.
echo "Downloading data..."
tmux new-session -s dl -n bash -d
tmux send-keys -t dl:0 'wget "https://www.cs.helsinki.fi/group/carat/data-sharing/carat-data-top1k-users-2014-to-2018-08-25.zip"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_details.csv"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/ui_layout_vectors.zip"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/traces.tar.gz"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/animations.tar.gz"' C-m
tmux send-keys -t dl:0 'wget "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/app_details.csv"' C-m
tmux send-keys -t dl:0 'wget "https://storage.cloud.google.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/rico_dataset_v0.1_semantic_annotations.zip"' C-m
## error handling
if [ $? == 0 ]; then
    echo "Download success!"
else
    echo "Error in downloading!"
fi