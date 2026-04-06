#!/bin/bash

# 1. Define the path to your other repository
PREFIX_REPO=~/FRNN-master/external/prefix_sum-master

# 2. Run the NVIDIA Compiler (nvcc)
# -I tells the compiler: "Look in this folder for header (.h) files"
nvcc -O3 -std=c++14 \
    -I./frnn/csrc/grid \
    -I./frnn/csrc/utils \
    -I$PREFIX_REPO \
    -I$PREFIX_REPO/parallel-scan \
    Main.cpp \
    frnn/csrc/grid/insert_points.cu \
    frnn/csrc/grid/find_nbrs.cu \
    $PREFIX_REPO/prefix_sum.cu \
    $PREFIX_REPO/parallel-scan/scan.cu \
    -o frnn_standalone

# 3. Check if it worked
if [ $? -eq 0 ]; then
    echo "Successfully compiled frnn_standalone!"
else
    echo "Compilation failed. Check the errors above."
fi
