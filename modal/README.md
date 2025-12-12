# Modal Deployment

## Use `vllm` PyPI package

Follow the official Modal [documentation](https://modal.com/docs/examples/vllm_inference) on deploying OpenAI-compatible LLM service with vLLM. Make the following changes:

- Change the `MODEL_NAME` and `MODEL_REVISION` to the latest LFM model. E.g. [`LFM2-8B-A1B`](https://huggingface.co/LiquidAI/LFM2-8B-A1B).
- Optionally, turn off `FAST_BOOT`.
- Optionally, add these environment variables:
  - `HF_XET_HIGH_PERFORMANCE=1`,
  - `VLLM_USE_V1=1`,
  - `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0`.
- Optionally, add these launch arguments:
  - `--dtype bfloat16`
  - `--gpu-memory-utilization 0.6`
  - `--max-model-len 32768`
  - `--max-num-seqs 600`
  - `--compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'`
- Collectively, these optional changes help to reduce the cold start time and boost the performance of LFM models on GPU instances.

The full script can be found in [`deploy-vllm.py`](./deploy-vllm.py).

Run `modal deploy vllm.py` to deploy the service.

Test the service with the following `curl` command (replace `<modal-deployment-url>` with your actual deployment URL):

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
