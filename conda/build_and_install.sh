#!/bin/bash
# Execute this script at upper dicretory to ignore build directory from commiting in git
rm -rf ./build/
conda build --output-folder ./build/ .
conda install hypycube -c ./build -y
