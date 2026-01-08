import os

import modal

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
print(f"Running deployment script for model: {MODEL_NAME}")

STARTUP_TIMEOUT_SECONDS = 400
CONTAINER_PORT = 8000
CONCURRENCY = 50
GPU_MEMORY_UTILIZATION = 0.6

app = modal.App(name="lfm-vllm-docker-inference")

vllm_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.12.0",
        secret=modal.Secret.from_name("dockerhub"),
    )
    .run_commands(
        "ln -sf /usr/bin/python3.12 /usr/local/bin/python",
        "ln -sf /usr/bin/python3.12 /usr/local/bin/python3",
    )
    .dockerfile_commands(
        # Clear the entrypoint. Reference:
        # https://modal.com/docs/guide/existing-images#entrypoint
        "ENTRYPOINT []",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            "MODEL_NAME": MODEL_NAME,
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    cpu=4,
    memory=16 * 1024,
    gpu="H100",
    scaledown_window=300,
    timeout=600,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    # Uncomment for production deployments
    # min_containers=1,
    # buffer_containers=1,
)
@modal.concurrent(max_inputs=CONCURRENCY, target_inputs=int(CONCURRENCY / 2))
@modal.web_server(CONTAINER_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve():
    import subprocess
    import time

    import requests

    # Launch vLLM
    print(f"Launching vLLM for model {MODEL_NAME} on port {CONTAINER_PORT}...")
    args = [
        f"--port {CONTAINER_PORT}",
        f"--model {MODEL_NAME}",
        "--tensor-parallel-size 1",
        "--dtype bfloat16",
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION}",
        "--max-model-len 32768",
        "--max-num-seqs 600",
        '--compilation-config \'{"cudagraph_mode": "FULL_AND_PIECEWISE"}\'',
    ]
    cmd = f"python -m vllm.entrypoints.openai.api_server {' '.join(args)}"
    print(f"Command: {cmd}")
    proc = subprocess.Popen(cmd, shell=True)

    # Wait for vLLM to become healthy (optional, only needed for warmup)
    max_wait = STARTUP_TIMEOUT_SECONDS - 10
    start_time = time.time()
    print(
        f"Waiting for model {MODEL_NAME} to launch on port {CONTAINER_PORT} at: {start_time}"
    )
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"http://localhost:{CONTAINER_PORT}/health", timeout=5
            )
            if response.status_code == 200:
                print("Application is healthy!")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    else:
        print(f"Model {MODEL_NAME} failed to become healthy in time")
        proc.kill()
        raise TimeoutError("Health check timeout")
    print(
        f"vLLM for model {MODEL_NAME} launched in {time.time() - start_time:.2f} seconds"
    )

    # Warmup vLLM (optional)
    print(f"Warming up model {MODEL_NAME} on port {CONTAINER_PORT}...")
    try:
        response = requests.post(
            f"http://localhost:{CONTAINER_PORT}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        content = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "N/A")
        )
        print(f"Warmup complete for {MODEL_NAME}. Response: {content}")
    except Exception as e:
        print(f"Warning: Warmup failed for {MODEL_NAME}: {str(e)}")
        print("Continuing startup despite warmup failure...")
