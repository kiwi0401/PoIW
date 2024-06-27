from typing import List, Optional
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import *


class Model(object):
    def __init__(self, model_path, device):
        self.device = device
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_path)

    def _load_model_and_tokenizer(self, model_path):
        logging.info(f"loading {model_path} model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, padding_side="left"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def generate_with_pow(
        self,
        query_texts: Optional[List[str]] = None,
        input_ids: Optional[dict] = None,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        prob_dist_cutoff_k: int = 5,
    ):
        """
        generate output from the model with proof of work
        here proof of work is top 5 tokens with their probabilities

        params:
            - query_texts: list of query texts for the model
            - max_new_tokens: maximum number of new tokens to generate
            - top_p: top_p nucleus for generation
            - probability_dist_cutoff_k: the cutoff index k for the top k
                                         tokens with the highest probabilities
                                         at each generation step.
        """
        self.model.eval()

        assert (
            query_texts is not None or input_ids is not None
        ), "one of query texts or input_ids must be given"

        if query_texts is not None:
            # tokenize the query texts
            input_ids = self.tokenizer(
                query_texts, return_tensors="pt", padding=True, truncation=False
            ).to(self.device)

        # generate up to `max_new_tokens` new tokens
        outputs = self.model.generate(
            input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=(top_p is not None),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # get output text
        sampled_token_ids = outputs.sequences
        sampled_tokens = self.tokenizer.batch_decode(
            sampled_token_ids, skip_special_tokens=True
        )

        # get topk probability dist (proof of work)
        all_logits = torch.stack(outputs.scores, dim=1)
        output_probs = F.softmax(all_logits, dim=-1)
        print(output_probs.shape)
        output_probs, output_prob_indices = torch.topk(
            output_probs, prob_dist_cutoff_k, dim=-1
        )

        return sampled_tokens, sampled_token_ids, output_probs, output_prob_indices

    def verify_responses(
        self,
        responses,
        n_tokens_to_check=3,
        rejection_threshold=1e-3,
    ):

        sampled_tokens, sampled_token_ids, output_probs, output_prob_indices = responses
        max_input_len = sampled_token_ids.shape[1] - output_probs.shape[1]
        prob_dist_cutoff_k = output_probs.shape[2]

        total_mse = 0.0

        # check random k token's prob
        for token_id in tqdm(
            np.random.choice(
                range(max_input_len, sampled_token_ids.shape[1]),
                n_tokens_to_check,
                replace=False,
            ),
            desc="verifying",
        ):
            _, _, topk_probs, topk_prob_indices = self.generate_with_pow(
                input_ids={
                    "input_ids": sampled_token_ids[:, :token_id],
                    "attention_mask": self.tokenizer.pad_token_id
                    != sampled_token_ids[:, :token_id],
                },
                max_new_tokens=1,
                prob_dist_cutoff_k=prob_dist_cutoff_k,
                top_p=None,  # no sampling is happening
            )

            # Squeeze to ensure 2D tensor for simpler operations
            masked_probs = (
                output_probs[:, token_id - max_input_len, :].squeeze().clone()
            )
            output_prob_indices_token = output_prob_indices[
                :, token_id - max_input_len, :
            ].squeeze()
            topk_probs = topk_probs.squeeze()
            topk_prob_indices = topk_prob_indices.squeeze()

            # Element-wise comparison for mismatched indices
            mask = ~(output_prob_indices_token == topk_prob_indices)
            masked_probs[mask] = 0.0  # Zero out

            total_mse += torch.mean((masked_probs - topk_probs) ** 2, dim=1)
            # print(total_mse)
        mse_loss = total_mse / n_tokens_to_check

        print(mse_loss <= rejection_threshold)
        return mse_loss <= rejection_threshold
