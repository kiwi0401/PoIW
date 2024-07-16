# install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# initialize conda
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
source ~/.bashrc
source ~/.zshrc

# install vllm
conda create --name vllm python=3.10
conda activate vllm
pip install vllm==0.5.1

# clone repo
git clone https://github.com/Neehan/PoIW.git
cd PoIW/
mkdir data data/llm_cache data/datasets
echo "HF_HOME=~/PoIW/data/llm_cache/" > .env

wget -P data/datasets/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python vllm/throughput_benchmark.py \
--model NousResearch/Meta-LLama-3-70B-Instruct \
--gpu-memory-utilization 0.98 \
-tp 8 \
--enforce-eager \

