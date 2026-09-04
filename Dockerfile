# syntax=docker/dockerfile:1.7

# CUDA 12.4.1, cuDNN and the complete compiler toolchain are required for
# nvdiffrast, PyTorch3D and Step1X-3D's custom rasterizer.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4
FROM ${CUDA_BASE_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
# RTX 4060 Ti (Ada Lovelace) uses CUDA compute capability 8.9.
ARG TORCH_CUDA_ARCH_LIST="8.9"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/models/huggingface \
    GRADIO_ANALYTICS_ENABLED=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV PATH=${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libegl1 \
        libeigen3-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        ninja-build \
        pkg-config \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv ${VIRTUAL_ENV} \
    && python -m pip install \
        pip==25.1.1 \
        setuptools==75.8.0 \
        wheel==0.45.1 \
        ninja==1.11.1.3 \
        packaging==24.2

WORKDIR /workspace/Step1X-3D

# Install the official CUDA 12.4 PyTorch wheels before packages which compile
# native CUDA extensions against PyTorch.
COPY requirements.txt requirements.cuda124.txt ./
RUN python -m pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-build-isolation -r requirements.txt \
    && python -m pip install -r requirements.cuda124.txt

COPY requirements.web.txt ./
RUN python -m pip install -r requirements.web.txt

# Keep native extension builds independent from changes to the Python app and
# pipelines. This makes later code-only rebuilds reuse the expensive sm_89
# compilation layer.
COPY step1x3d_texture/custom_rasterizer ./step1x3d_texture/custom_rasterizer
COPY step1x3d_texture/differentiable_renderer ./step1x3d_texture/differentiable_renderer
RUN python -m pip install --no-build-isolation ./step1x3d_texture/custom_rasterizer \
    && python -m pip install --no-build-isolation ./step1x3d_texture/differentiable_renderer

COPY . .

# Fail the build if Python, PyTorch or their CUDA ABI do not match the setup
# against which the model and native extensions are expected to run.
RUN python -m pip check \
    && python - <<'PY'
import sys
import torch
import kaolin
import nvdiffrast.torch
import pytorch3d
import torch_cluster
import custom_rasterizer
import mesh_processor
import gradio as gr
import pydantic

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split("+")[0] == "2.5.1", torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert pydantic.__version__ == "2.10.6", pydantic.__version__

# Gradio 5.5 fails at runtime if its client cannot parse a component's JSON
# schema. Exercise the same Image -> Model3D combination used by app.py.
with gr.Blocks() as schema_test:
    schema_input = gr.Image(type="filepath")
    schema_output = gr.Model3D()
    gr.Button().click(lambda image: image, schema_input, schema_output)
assert schema_test.get_api_info()["named_endpoints"]

print(f"Python {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}, CUDA ABI {torch.version.cuda}")
print(f"Pydantic {pydantic.__version__}, Gradio API schema OK")
print("Native CUDA/C++ extensions imported successfully")
PY

RUN groupadd --gid ${GROUP_ID} step1x \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home step1x \
    && mkdir -p /models/huggingface cache output /home/step1x/.u2net \
    && chown -R step1x:step1x /models /workspace/Step1X-3D /home/step1x

USER step1x

EXPOSE 7861

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7861"]
