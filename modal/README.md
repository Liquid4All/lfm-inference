# Modal Deployment

## Use `vllm` PyPI package

- Follow the official Modal [documentation](https://modal.com/docs/examples/vllm_inference) on deploying OpenAI-compatible LLM service with vLLM.
- Change the `MODEL_NAME` and `MODEL_REVISION` to the latest LFM model. E.g. `https://huggingface.co/LiquidAI/LFM2-8B-A1B`.
- Run `modal deploy vllm.py` to deploy the service.
- Test the service:

```sh
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
