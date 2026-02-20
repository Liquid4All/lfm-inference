# LFM Inference

This repository contains examples of deploying inference servers for Liquid Foundation Model (LFM).

## Platform-specific deployment

- [Fal](./fal)
- [Baseten](./baseten)
- [Modal](./modal)

## General deployment

> [!NOTE]
> The VL model is not compatible with the public vLLM image yet. The deployment is for text model only for now.

- Use the latest public vLLM container, version `0.15.1`.
- Install `transformers` version `5.1.0`.
- Add these environment variables:

```
VLLM_USE_V1=1
VLLM_USE_FUSED_MOE_GROUPED_TOPK=0
```

- Launch vLLM with the following arguments:

```
--tensor-parallel-size 1
--dtype bfloat16
--gpu-memory-utilization 0.6
--max-model-len 32768
--max-num-seqs 600
```

Adjust these settings based on your hardware:

```
--gpu-memory-utilization
--max-num-seqs
--kv-cache-memory-bytes
```

## Recommended generation parameters

**Text models**

| Parameter | Value |
| --- | --- |
| `temperature` | 0.3 |
| `min_p` | 0.15 |
| `repetition_penalty` | 1.05 |

**Text instruct models**

| Parameter | Value |
| --- | --- |
| `temperature` | 0.1 |
| `top_p` | 0.1 |
| `top_k` | 50 |
| `repetition_penalty` | 1.05 |

**VL models**

| Parameter | Value |
| --- | --- |
| `temperature` | 0.1 |
| `min_p` | 0.15 |
| `repetition_penalty` | 1.05 |

**Nano models**

| Parameter | Value |
| --- | --- |
| `temperature` | 0.0 |
| `min_p` | 1.0 |

## Local development

Lint and format code:

```
uv sync
uv run ruff check . --fix
uv run ruff format .
```

## License

[MIT](./LICENSE)
