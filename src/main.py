import random
import numpy as np
import torch
import argparse
import json

import model as m
import utils
from constants import *

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="pythia-70m-deduped", help="model name"
    )
    parser.add_argument("--data-dir", type=str, default="data/", help="data directory")
    parser.add_argument(
        "--verifier-input",
        type=str,
        default="pythia-70m-deduped_output.json",
        help="verifier input file",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="max new tokens"
    )
    parser.add_argument(
        "--prob-cutoff-k", type=int, default=10, help="probability dist cutoff k"
    )
    parser.add_argument(
        "--n-audit-tokens", type=int, default=5, help="number of tokens to audit"
    )
    parser.add_argument(
        "--rejection-threshold",
        type=float,
        default=0.005,
        help="threshold for rejection",
    )
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        help="generate output (stores output in data dir)",
    )
    parser.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        help="verify output (read from data dir)",
    )
    parser.set_defaults(generate=False, verify=False)

    args = parser.parse_args()
    assert (
        args.generate ^ args.verify
    ), "must set exactly one of generate and verify flags"

    model = m.Model(args.data_dir + "llm_cache/" + args.model, device=DEVICE)

    if args.generate:
        logging.info("generating output")
        output = []
        for query_texts in utils.iterate_human_eval(
            split="test", batch_size=args.batch_size
        ):
            sampled_tokens, sampled_token_ids, output_probs, output_prob_indices = (
                model.generate_with_pow(
                    query_texts=query_texts,
                    max_new_tokens=args.max_new_tokens,
                    prob_dist_cutoff_k=args.prob_cutoff_k,
                )
            )

            output.append(
                {
                    "query_texts": query_texts,
                    "response": {
                        "sampled_tokens": sampled_tokens,
                        "sampled_token_ids": sampled_token_ids.cpu().tolist(),
                        "output_probs": output_probs.cpu().tolist(),
                        "output_prob_indices": output_prob_indices.cpu().tolist(),
                    },
                }
            )

        logging.info("saving output")
        with open(args.data_dir + f"{args.model}_output.json", "w") as f:
            json.dump(output, f)
            f.close()
    if args.verify:
        total = 0.0
        accepted = 0.0
        for responses in utils.read_responses(args.data_dir + args.verifier_input):
            decisions = model.verify_responses(
                responses,
                n_tokens_to_check=args.n_audit_tokens,
                rejection_threshold=args.rejection_threshold,
            )
            total += len(decisions)
            accepted += decisions.sum()

        print(
            f"# total: {total}, # accepted: {accepted}, pct: {accepted / total * 100 : .2f}%"
        )
