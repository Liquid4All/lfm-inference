# LFM Inference

This repository contains examples of deploying inference servers for Liquid Foundation Model (LFM).

## Platform-specific deployment

- [Fal](./fal)
- [Baseten](./baseten)
- [Modal](./modal)

## General deployment

> [!NOTE]
> The VL model is not compatible with the public vLLM image yet. The deployment is for text model only for now.

1. Use the latest public vLLM container.
2. Add these environment variables:

```
VLLM_USE_V1=1
VLLM_USE_FUSED_MOE_GROUPED_TOPK=0
```

3. Launch vLLM with the following arguments:

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
--kv-cache-memory-bytes
```

## License

[MIT](./LICENSE)
