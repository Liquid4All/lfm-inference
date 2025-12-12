import modal
import os


MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
print(f"Deploying model: {MODEL_NAME}")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_USE_V1": "1",
        "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "0",
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lfm-vllm-pypi-inference")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # Uncomment for production deployments
    # min_containers=1,
    # buffer_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        f"--served-model-name {MODEL_NAME}",
        "--host 0.0.0.0",
        f"--port {str(VLLM_PORT)}",
        # extra arguments
        '--dtype bfloat16',
        '--gpu-memory-utilization 0.6',
        '--max-model-len 32768',
        '--max-num-seqs 600',
        "--compilation-config '{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'",
    ]

    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
