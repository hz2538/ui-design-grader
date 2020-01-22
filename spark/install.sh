#!/bin/bash

## install needed python packages
echo "Installing python packages..."
sudo apt-get install python3-pip -y
# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# enter "yes", press enter to go through, and enter "yes" in the end
source ~/.bashrc
conda install pandas numpy scipy pillow matplotlib scikit-learn
# YOU CAN SKIP THE JUPYTER NOTEBOOK INSTALLATION
# conda install jupyter notebook
# jupyter notebook --generate-config
# vi ~/.jupyter/jupyter_notebook_config.py
# ADD THOSE LINES IN THE BOTTOM OF THE FILE
## c = get_config()
## c.NotebookApp.ip='*'
## c.NotebookApp.open_browser = False
## c.NotebookApp.port =9999      # or other port number
# jupyter notebook password
sudo pip3 install pyspark nose h5py py4j boto3 s3fs sparkdl dill sparkflow

## error handling
if [ $? == 0 ]; then
    echo "Install success!"
else
    echo "Error in installation!"
fi