from datasets import load_dataset
from tqdm import tqdm
import torch
from constants import *
import json


def iterate_human_eval(data_dir, split="test", batch_size=16):
    dataset = load_dataset(
        data_dir + "datasets/openai_humaneval", trust_remote_code=True
    )
    texts = [example["prompt"] for example in dataset[split]]
    for i in tqdm(range(0, len(texts), batch_size), desc="HumanEval"):
        yield texts[i : i + batch_size]


def read_responses(filepath):
    with open(filepath, "r") as f:
        output = json.load(f)
        f.close()
    for data_point in output:
        sampled_tokens = data_point["response"]["sampled_tokens"]
        sampled_token_ids = torch.tensor(
            data_point["response"]["sampled_token_ids"],
            dtype=torch.long,
            device=DEVICE,
        )
        output_probs = torch.tensor(
            data_point["response"]["output_probs"],
            dtype=torch.float32,
            device=DEVICE,
        )
        output_prob_indices = torch.tensor(
            data_point["response"]["output_prob_indices"],
            dtype=torch.long,
            device=DEVICE,
        )
        responses = (
            sampled_tokens,
            sampled_token_ids,
            output_probs,
            output_prob_indices,
        )
        yield responses
