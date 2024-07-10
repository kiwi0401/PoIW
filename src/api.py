from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from model import Model
import numpy as np
import torch

app = FastAPI()

provider_model = Model("data/llm_cache/pythia-160m-deduped", device="cpu")


async def generate_tokens(user_prompt, system_prompt, max_tokens, seed):
    random.seed(seed)  # Set the random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    prompt = f"{user_prompt}"
    (
        sampled_tokens,
        _,
        _,
        _,
    ) = provider_model.generate_with_pow(
        [prompt], max_new_tokens=max_tokens, prob_dist_cutoff_k=10
    )
    print(sampled_tokens)

    sampled_tokens = sampled_tokens[0]

    yield sampled_tokens


@app.get("/stream")
async def stream_response(
    user_prompt: str = Query(...),
    system_prompt: str = Query(...),
    seed: int = Query(...),
    max_tokens: int = Query(...),
    provider_type: str = Query(...),
):
    # Validate the seed parameter
    if not isinstance(seed, int):
        raise HTTPException(status_code=400, detail="Seed must be an integer")

    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be an integer")

    if provider_type.lower() != "honest" and provider_type.lower() != "dishonest":
        raise HTTPException(
            status_code=400,
            detail="Unknown GPU provider type. Must be `honest` or `dishonest`",
        )

    # Create a generator function with parameters
    data_generator = generate_tokens(user_prompt, system_prompt, max_tokens, seed)

    # Stream response using StreamingResponse
    return StreamingResponse(data_generator, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    import random

    uvicorn.run(app, host="0.0.0.0", port=8000)
