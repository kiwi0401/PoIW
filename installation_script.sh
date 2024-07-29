#!/bin/bash

# Update package lists and install dependencies
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs nginx

# Install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize conda
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
source ~/.bashrc
source ~/.zshrc

# Create vllm environment and install vllm
conda create --name vllm python=3.10 -y
conda activate vllm && pip install vllm==0.5.3.post1

# Clone PoIW repo and set up directories
git clone https://github.com/Neehan/PoIW.git
cd PoIW/
mkdir -p data data/llm_cache data/datasets
echo "HF_HOME=~/PoIW/data/llm_cache/" > .env

# Download dataset
wget -P data/datasets/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Update and install nginx
sudo apt update
sudo apt install nginx -y
mv vllm_server/nginx.text /etc/nginx/sites-available/default

# Set up environment variables
env >> /etc/environment

# Install the latest preview version of TensorRT-LLM
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Verify the installation of TensorRT-LLM
python3 -c "import tensorrt_llm"

# Clone the TensorRT-LLM repository and install example requirements
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install -r examples/bloom/requirements.txt
git lfs install

echo "Installation complete."

# Run the server on port 9000, nginx on port 8000, and forward traffic
# Uncomment the following lines if you wish to run the server immediately
# python -m vllm_server.server --host 127.0.0.1 --port 9000 --download-dir ~/PoIW/data/llm_cache/ --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8
