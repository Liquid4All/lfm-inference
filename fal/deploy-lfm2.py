import os
import subprocess
import sys

import fal
from fal import ContainerImage

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
PY_VERSION = ".".join(str(x) for x in sys.version_info[:2])

DOCKERFILE_STR = f"""
FROM vllm/vllm-openai:v0.15.1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv v --seed --python={PY_VERSION} /workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"
RUN uv pip install transformers==5.1.0

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VLLM_USE_V1=1
"""


@fal.function(
    kind="container",
    exposed_port=8000,
    image=ContainerImage.from_dockerfile_str(DOCKERFILE_STR),
    machine_type="GPU-H100",
    num_gpus=1,
)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.6",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "600",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_AND_PIECEWISE"}',
    ]
    subprocess.run(cmd)
