#!/bin/bash

## install needed python packages
echo "Installing python packages..."
conda install pyspark numpy nose pillow h5py py4j boto3 s3fs pandas dill
## error handling
if [ $? == 0 ]; then
    echo "Install success!"
else
    echo "Error in installation!"
fi