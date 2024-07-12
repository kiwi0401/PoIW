from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

from utils import to_json


def main():

    parser = FlexibleArgumentParser(description="VLLM Benchmark")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="rubra-ai/Meta-Llama-3-8B-Instruct",
        help="model name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="data dir",
    )

    parser.add_argument(
        "--max-tokens", type=int, default=1, help="maximum output tokens"
    )

    parser.add_argument(
        "--num-logprobs",
        "-n",
        type=int,
        default=20,
        help="number of log probs per output token",
    )

    parser.add_argument(
        "--temperature", "-t", type=float, default=0.0, help="temperature"
    )

    args = parser.parse_args()

    model = LLM(
        args.model,
        download_dir=args.data_dir + "/llm_cache/",
        tensor_parallel_size=1,  # parallelize model weights to 1 gpu
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=1234,
        logprobs=args.num_logprobs,
    )
    outputs = model.generate("Hello my name is", sampling_params=sampling_params)
    print(to_json(outputs))


if __name__ == "__main__":
    main()
