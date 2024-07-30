# copied from https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py

"""Benchmark offline inference throughput."""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import tensorrt_llm
import torch
from huggingface_hub import login
from tensorrt_llm._utils import release_gc
from tensorrt_llm.bindings.executor import Executor, Request, SamplingConfig
from tensorrt_llm.bindings.executor import ModelType, ExecutorConfig  # Ensure correct import
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.llama.convert import load_hf_llama
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser


def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def run_vllm(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer: str,
        quantization: Optional[str],
        tensor_parallel_size: int,
        seed: int,
        n: int,
        use_beam_search: bool,
        trust_remote_code: bool,
        dtype: str,
        max_model_len: Optional[int],
        enforce_eager: bool,
        kv_cache_dtype: str,
        quantization_param_path: Optional[str],
        device: str,
        enable_prefix_caching: bool,
        enable_chunked_prefill: bool,
        max_num_batched_tokens: int,
        distributed_executor_backend: Optional[str],
        gpu_memory_utilization: float = 0.9,
        download_dir: Optional[str] = None,
        load_format: str = EngineArgs.load_format,
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
    )

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []

    total_time = 0

    for i, request in enumerate(requests):
        prompt, _, output_len = request
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            )
        )

        # batch 5 at a time
        # if len(prompts) == 5 or i == len(requests) - 1:
        #     start = time.perf_counter()
        #     llm.generate(prompts, sampling_params, use_tqdm=True)
        #     end = time.perf_counter()
        #     total_time += end - start

        #     prompts = []
        #     sampling_params = []

    # return total_time  # end - start

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    return end - start


def run_hf(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        n: int,
        use_beam_search: bool,
        max_batch_size: int,
        trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
    )
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (
                    max(max_prompt_len, next_prompt_len)
                    + max(max_output_len, next_output_len)
            ) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
        requests: List[Tuple[str, int, int]],
        model: str,
        tensor_parallel_size: int,
        output_len: int,
) -> float:
    from mii import client, serve

    llm = serve(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm.generate(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    client = client(model)
    client.terminate_server()
    return end - start


def build_engine(output_dir, max_batch_size):
    from tensorrt_llm import BuildConfig
    from tensorrt_llm.models import LLaMAForCausalLM

    print("Starting TensorRTLLM engine build...")

    llama = LLaMAForCausalLM.from_checkpoint(output_dir)
    build_config = BuildConfig(max_batch_size=max_batch_size)
    engine = tensorrt_llm.build(llama, build_config)
    engine.save(os.path.join(output_dir, "engine.trt"))

    print("Engine build completed.")

    return engine


def convert_hf_model_to_trtllm(model_dir, output_dir, dtype, load_model_on_cpu, load_by_shard, tp_size, pp_size):
    print("Starting model conversion...")

    world_size = tp_size * pp_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except:
        print("AutoConfig cannot load the Hugging Face config.")

    hf_model = load_hf_llama(model_dir, load_model_on_cpu)

    def convert_and_save_rank(rank):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size,
            moe_tp_size=1,
            moe_ep_size=1
        )
        llama = LLaMAForCausalLM.from_hugging_face(
            model_dir if hf_model is None else hf_model,
            dtype,
            mapping=mapping,
            quant_config=None,
            load_by_shard=load_by_shard,
        )
        llama.save_checkpoint(output_dir, save_config=(rank == 0))
        del llama

    with ThreadPoolExecutor(max_workers=world_size) as executor:
        futures = [executor.submit(convert_and_save_rank, rank) for rank in range(world_size)]
        for future in as_completed(futures):
            future.result()
    release_gc()

    print("Model conversion completed.")


def run_trtllm(requests, model_dir, trt_output_dir, dtype):
    # Login for closed models
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("HUGGINGFACE_TOKEN environment variable not set")
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
    login(hf_token)

    # Convert the model and empty vcache to make space for optimized model
    convert_hf_model_to_trtllm(model_dir, trt_output_dir, dtype, False, False, 1, 1)
    torch.cuda.empty_cache()

    # Build engine & Tokenizer
    engine = build_engine(trt_output_dir, args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create executor and load model to GPU
    executor = Executor(model_path=os.path.join(trt_output_dir, "engine.trt"), model_type=ModelType.DECODER_ONLY,
                        executor_config=ExecutorConfig())
    print("Executor initialized.")

    prompts = [prompt for prompt, _, _ in requests]
    start = time.perf_counter()
    results = []
    requests = []
    total_responses = 0
    # Create and enqueue requests
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].tolist()
        sampling_config = SamplingConfig()
        request = Request(input_token_ids=input_ids, streaming=False,
                          sampling_config=sampling_config)
        requests.append(request)

        request_amount = 10  # Number of requests to process at a time
        batch_counter = 0
        responses_received = 0
        if len(requests) >= request_amount:
            print("EXECUTING REQUESTS")
            print(len(requests))
            executor.enqueue_requests(requests)

            # Wait for responses

            while responses_received < request_amount:
                responses = executor.await_responses()
                for response in responses:
                    if response.has_error():
                        print(f"Error in response {response.request_id}: {response.error_msg}")
                    else:
                        print("NUM RESPONSES")
                        print(total_responses)
                        print("-" * 80)
                    responses_received += 1
                    total_responses += 1

            requests = []
            batch_counter += 1

    end = time.perf_counter()

    executor.shutdown()

    print("Inference completed.")

    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [
            (prompt, args.input_len, args.output_len) for _ in range(args.num_prompts)
        ]
    else:
        requests = sample_requests(
            args.dataset, args.num_prompts, tokenizer, args.output_len
        )

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests,
            args.model,
            args.tokenizer,
            args.quantization,
            args.tensor_parallel_size,
            args.seed,
            args.n,
            args.use_beam_search,
            args.trust_remote_code,
            args.dtype,
            args.max_model_len,
            args.enforce_eager,
            args.kv_cache_dtype,
            args.quantization_param_path,
            args.device,
            args.enable_prefix_caching,
            args.enable_chunked_prefill,
            args.max_num_batched_tokens,
            args.distributed_executor_backend,
            args.gpu_memory_utilization,
            args.download_dir,
            args.load_format,
        )
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(
            requests,
            args.model,
            tokenizer,
            args.n,
            args.use_beam_search,
            args.hf_max_batch_size,
            args.trust_remote_code,
        )
    elif args.backend == "mii":
        elapsed_time = run_mii(
            requests, args.model, args.tensor_parallel_size, args.output_len
        )
    elif args.backend == "tensorrt":
        elapsed_time = run_trtllm(
            requests, args.model, args.engine_dir, args.dtype
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len for _, prompt_len, output_len in requests
    )
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "hf", "mii", "tensorrt"], default="vllm"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
             "output length from the dataset.",
    )
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--quantization", "-q", choices=[*QUANTIZATION_METHODS, None], default=None
    )
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum length of a sequence (including prompt and output). "
             "If None, will be derived from the model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="data type for model weights and activations. "
             'The "auto" option will use FP16 precision '
             "for FP32 and FP16 models, and BF16 precision "
             "for BF16 models.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="the fraction of GPU memory to be used for "
             "the model executor, which can range from 0 to 1."
             "If unspecified, will use the default value of 0.9.",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="enforce eager execution"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
             "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
             "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)",
    )
    parser.add_argument(
        "--quantization-param-path",
        type=str,
        default=None,
        help="Path to the JSON file containing the KV cache scaling factors. "
             "This should generally be supplied, when KV cache dtype is FP8. "
             "Otherwise, KV cache scaling factors default to 1.0, which may cause "
             "accuracy issues. FP8_E5M2 (without scaling) is only supported on "
             "cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is "
             "instead supported for common inference criteria.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help="device type for vLLM execution, supporting CUDA, OpenVINO and " "CPU.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="enable automatic prefix caching for vLLM backend.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="enable chunked prefill for vLLM backend.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="maximum number of batched tokens per " "iteration",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/llm_cache/",
        help="directory to download and load the weights, "
             "default to the default cache dir of huggingface",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument('--engine_dir', type=str, default='/code/tensorrt_llm/mistral_trtllm/llama_style_merge_long_v2',help="Path to the TensorRT Engine directory.")
    parser.add_argument(
        "--distributed-executor-backend",
        choices=["ray", "mp"],
        default=None,
        help="Backend to use for distributed serving. When more than 1 GPU "
             'is used, will be automatically set to "ray" if installed '
             'or "mp" (multiprocessing) otherwise.',
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default=EngineArgs.load_format,
        choices=[
            "auto",
            "pt",
            "safetensors",
            "npcache",
            "dummy",
            "tensorizer",
            "bitsandbytes",
        ],
        help="The format of the model weights to load.\n\n"
             '* "auto" will try to load the weights in the safetensors format '
             "and fall back to the pytorch bin format if safetensors format "
             "is not available.\n"
             '* "pt" will load the weights in the pytorch bin format.\n'
             '* "safetensors" will load the weights in the safetensors format.\n'
             '* "npcache" will load the weights in pytorch format and store '
             "a numpy cache to speed up the loading.\n"
             '* "dummy" will initialize the weights with random values, '
             "which is mainly for profiling.\n"
             '* "tensorizer" will load the weights using tensorizer from '
             "CoreWeave. See the Tensorize vLLM Model script in the Examples"
             "section for more information.\n"
             '* "bitsandbytes" will load the weights using bitsandbytes '
             "quantization.\n",
    )
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError(
                "Tokenizer must be the same as the model for MII " "backend."
            )
    main(args)
