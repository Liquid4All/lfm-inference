# Modal Deployment

## Install

```sh
cd modal
pip install modal
```

## Use `vLLM` docker image

Use the pre-built `vLLM` docker image `vllm/vllm-openai` to deploy LFM.

Launch command:

```sh
# deploy LFM2 8B MoE model
modal deploy deploy-vllm-docker.py

# deploy other LFM2 model, MODEL_NAME defaults to LiquidAI/LFM2-8B-A1B
MODEL_NAME=LiquidAI/<model-slug> modal deploy deploy-vllm-docker.py
```

See full list of open source LFM models on [Hugging Face](https://huggingface.co/collections/LiquidAI/lfm2).

## Use `vLLM` with sleep mode

Launch command:

```sh
# deploy LFM2 8B MoE model
modal deploy deploy-vllm-with-sleep.py

# deploy any LFM2 model, MODEL_NAME defaults to LiquidAI/LFM2-8B-A1B
MODEL_NAME=LiquidAI/<model-slug> modal deploy deploy-vllm-with-sleep.py
```

> [!NOTE]
> This deployment enables both CPU and GPU memory snapshots. The first cold start takes about 3.5 - 5 min, which is longer than the time without the snapshot. But **subsequent cold starts are much faster, in 0.5 - 1 min**.

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
