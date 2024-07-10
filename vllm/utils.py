def to_json(responses):
    outputs = []
    for response in responses:
        outputs.append(
            {
                "prompt": response.prompt,
                "output": response.outputs[0].text,
                "logprobs": [
                    [
                        {
                            "logprob": logprob.logprob,
                            "rank": logprob.rank,
                            "decoded_token": logprob.decoded_token,
                        }
                        for logprob in logprobs.values()
                    ]
                    for logprobs in response.outputs[0].logprobs
                ],
            }
        )
    return outputs
