#!/bin/bash

## install needed flask packages
echo "Installing flask packages..."
pip install flask flask-uploads flask-dropzone psycopg2 imutils
## error handling
if [ $? == 0 ]; then
    echo "Install success!"
else
    echo "Error in installation!"
fi