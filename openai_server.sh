python -m vllm.entrypoints.openai.api_server \
--model rubra-ai/Meta-Llama-3-8B-Instruct \
--dtype auto \
--download-dir data/llm_cache/ \
--port 8080 \
# --api-key token-abc123 \
