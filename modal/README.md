# Modal Deployment

## Use `vLLM` PyPI package

Follow the official Modal [documentation](https://modal.com/docs/examples/vllm_inference) on deploying OpenAI-compatible LLM service with vLLM. Make the following changes:

- Change the `MODEL_NAME` and `MODEL_REVISION` to the latest LFM model.
  - E.g. for[`LFM2-8B-A1B`](https://huggingface.co/LiquidAI/LFM2-8B-A1B):
    - `MODEL_NAME = "LiquidAI/LFM2-8B-A1B"`
    - `MODEL_REVISION = "6df6a75822a5779f7bf4a21e765cb77d0383935d"`
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

Run `modal deploy deploy-vllm.py` to deploy the service.

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

> [!NOTE]
> Since vLLM takes over 2 min to cold start, if you run the inference server for production, it is recommended to keep a minimum number of warm instances with `min_containers = 1` and `buffer_containers = 1`. See [docs](https://modal.com/docs/guide/cold-start#overprovision-resources-with-min_containers-and-buffer_containers) for details.

## Use `vLLM` docker image

Alternatively, you can use the official `vLLM` docker image to deploy the LFM inference server.

Run `modal deploy deploy-vllm-docker.py` to deploy the service.

This approach provides better performance over the PyPI package approach, as the docker image is pre-built with optimizations for inference.

## Production deployment

- Prefer the `deploy-vllm-docker.py` script.
- Since vLLM takes over 2 min to cold start, if you run the inference server for production, it is recommended to keep a minimum number of warm instances with `min_containers = 1` and `buffer_containers = 1`. The `buffer_containers` config is necessary because all Modal GPUs are subject to [preemption](https://modal.com/docs/guide/preemption). See [docs](https://modal.com/docs/guide/cold-start#overprovision-resources-with-min_containers-and-buffer_containers) for details about cold start performance tuning.
- Warm up the vLLM server after deployment by sending a single request. The warm-up process is included in the [deploy-vllm-docker.py](./deploy-vllm-docker.py) script.
