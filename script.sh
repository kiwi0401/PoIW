#!/bin/bash
#SBATCH --gres=gpu:volta:1

# Loading the required module
module load anaconda/Python-ML-2023b

export HF_HOME=/state/partition1/user/$(whoami)/hf
mkdir -p $(HF_HOME)
mkdir -p $(HF_HOME)/datsets

cp -r /data/datasets $(HF_HOME)/datasets

# Run the script
python src/main.py "$@"
