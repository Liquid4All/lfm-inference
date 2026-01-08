# Modal Deployment

## Use `vLLM` PyPI package

Use the `vLLM` PyPI package to deploy LFM. This approach is based on the Modal [example](https://modal.com/docs/examples/vllm_inference) for deploying OpenAI-compatible LLM service with vLLM, with a few modifications.

Launch command:

```sh
cd modal

# deploy LFM2 8B MoE model
modal deploy deploy-vllm-pypi.py

# deploy any LFM2 model, MODEL_NAME defaults to LiquidAI/LFM2-8B-A1B
MODEL_NAME=LiquidAI/<model-slug> modal deploy deploy-vllm-pypi.py
```

> [!NOTE]
> This deployment enables both CPU and GPU memory snapshots. The first cold start takes about 3.5 - 5 min, which is longer than the time without the snapshot. But **subsequent cold starts are much faster, around 30 seconds**.

## Use `vLLM` docker image

Alternatively, you can use the pre-built `vLLM` docker image `vllm/vllm-openai` to deploy LFM.

Launch command:

```sh
cd modal

# deploy LFM2 8B MoE model
modal deploy deploy-vllm-docker.py

# deploy other LFM2 model, MODEL_NAME defaults to LiquidAI/LFM2-8B-A1B
MODEL_NAME=LiquidAI/<model-slug> modal deploy deploy-vllm-docker.py
```

See full list of open source LFM models on [Hugging Face](https://huggingface.co/collections/LiquidAI/lfm2).

## Test commands

Test the deployed server with the following `curl` commands (replace `<modal-deployment-url>` with your actual deployment URL):

```sh
# List deployed model
curl https://<modal-deployment-url>/v1/models

# Query the deployed LFM model
curl https://<modal-deployment-url>/v1/chat/completions \
  --json '{
  "model": "LiquidAI/LFM2-8B-A1B",
  "messages": [
    {
      "role": "user",
      "content": "What is the melting temperature of silver?"
    }
  ],
  "max_tokens": 32,
  "temperature": 0
}'
```
