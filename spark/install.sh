#!/bin/bash

## install needed python packages
echo "Installing python packages..."
sudo apt-get install python3-pip -y
sudo pip3 install pyspark numpy nose pillow h5py py4j boto3 s3fs sparkdl pandas dill sparkflow
## error handling
if [ $? == 0 ]; then
    echo "Install success!"
else
    echo "Error in installation!"
fi