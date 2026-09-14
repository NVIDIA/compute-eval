ARG CUDA_BASE_IMAGE=nvcr.io/nvidia/cuda:13.1.0-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE} AS solve-base

ARG DEBIAN_FRONTEND=noninteractive
ARG NODE_MAJOR=22
ARG UV_VERSION=0.9.13
ARG CUDNN_VERSION=9.20.0.48-1
ARG CUDNN_FRONTEND_VERSION=v1.23.0
# Nsight profilers (nsys/ncu) are installed by default. Set --build-arg INSTALL_NSIGHT=0
# to skip them, e.g. when profiling is done off-container via the Cloud Profiling
# Service or the base image already ships its own profiler build.
ARG INSTALL_NSIGHT=1
ARG NSIGHT_SYSTEMS_PACKAGE=nsight-systems-2025.5.2
ARG NSIGHT_COMPUTE_PACKAGE=nsight-compute-2025.4.0
ARG CLAUDE_CODE_VERSION=latest
ARG OPENCODE_VERSION=latest
# CUTLASS pins MUST match docker/Dockerfile.eval in the compute-eval grader repo,
# so agents build/iterate against the exact headers the grader validates against.
ARG CUTLASS_VERSION=4.6.1
ARG CUTLASS_GIT_SHA=e05f953a5b3d38adc240df2ff928e0421c2abba3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      openssh-client \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      ripgrep \
      unzip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "libcudnn9-headers-cuda-13=${CUDNN_VERSION}" \
      "libcudnn9-cuda-13=${CUDNN_VERSION}" \
      "libcudnn9-dev-cuda-13=${CUDNN_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${CUDNN_FRONTEND_VERSION}" https://github.com/NVIDIA/cudnn-frontend.git /tmp/cudnn-frontend \
    && cp -r /tmp/cudnn-frontend/include/* /usr/local/include/ \
    && rm -rf /tmp/cudnn-frontend

RUN if [ "${INSTALL_NSIGHT}" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends \
        "${NSIGHT_SYSTEMS_PACKAGE}" \
        "${NSIGHT_COMPUTE_PACKAGE}" \
      && rm -rf /var/lib/apt/lists/* \
      && command -v nsys \
      && command -v ncu; \
    fi

# CUTLASS C++ headers, pinned to the immutable commit the grader uses (never the
# moving main branch). Agents solving cutlass/* problems build with
# -I /opt/cutlass/include; without this they cannot compile their own solutions.
RUN git init /opt/cutlass && \
    git -C /opt/cutlass remote add origin https://github.com/NVIDIA/cutlass.git && \
    git -C /opt/cutlass fetch --depth 1 origin ${CUTLASS_GIT_SHA} && \
    git -C /opt/cutlass checkout --detach FETCH_HEAD && \
    test "$(git -C /opt/cutlass rev-parse HEAD)" = "${CUTLASS_GIT_SHA}"

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --only-binary=:all: \
      "uv==${UV_VERSION}" \
    && uv venv /opt/cuda-python-venv

RUN uv pip install --python /opt/cuda-python-venv/bin/python \
      --only-binary=:all: \
      cupy-cuda13x \
      cuda-tile \
      "nvidia-cutlass-dsl[cu13]==${CUTLASS_VERSION}"

RUN uv pip install --python /opt/cuda-python-venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch

RUN uv pip install --python /opt/cuda-python-venv/bin/python \
      --only-binary=:all: \
      cuda-python \
      matplotlib \
      numba \
      numpy \
      nvtx \
      pandas \
      pytest \
      pyyaml \
      scikit-learn \
      scipy

RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    CUTLASS_PATH=/opt/cutlass \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/opt/cuda-python-venv/bin:/usr/local/cuda/bin:${PATH} \
    UV_PYTHON=/opt/cuda-python-venv/bin/python

RUN if python -c "import compute_eval" 2>/dev/null; then \
      echo "compute_eval must not be installed in the Harbor solve image." >&2; \
      exit 1; \
    fi \
    && python -c "import cuda.tile; print('cuda.tile OK')" \
    && python -c "import torch; print(f'PyTorch {torch.__version__}')" \
    && python -c "import cupy; print(f'CuPy {cupy.__version__}')" \
    && python -c "import numpy; print(f'NumPy {numpy.__version__}')" \
    && printf '#include <cutlass/version.h>\nint main(){return (int)cutlass::getVersion();}\n' > /tmp/cutlass_smoke.cu \
    && nvcc -std=c++17 -I /opt/cutlass/include -c /tmp/cutlass_smoke.cu -o /tmp/cutlass_smoke.o \
    && rm /tmp/cutlass_smoke.cu /tmp/cutlass_smoke.o \
    && python -c "from importlib.metadata import version; import cutlass, cutlass.cute; v=version('nvidia-cutlass-dsl'); assert v=='${CUTLASS_VERSION}', v; print(f'CUTLASS DSL {v}')" \
    && which rg \
    && nvcc --version

RUN mkdir -p /workspace && chmod 1777 /workspace

WORKDIR /workspace

# HOME must be writable by the arbitrary UID Harbor exec's as (see agent.user
# in task.toml). Set it per leaf stage AFTER npm/version-check RUNs so root
# doesn't salt it with unwritable subdirs during build.
FROM solve-base AS claude-code

ARG CLAUDE_CODE_VERSION

RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version

ENV HOME=/opt/agent-home
RUN mkdir -p /opt/agent-home && chmod 1777 /opt/agent-home

CMD ["bash"]

FROM solve-base AS opencode

ARG OPENCODE_VERSION

RUN npm install -g "opencode-ai@${OPENCODE_VERSION}" \
    && opencode --version

ENV HOME=/opt/agent-home
RUN mkdir -p /opt/agent-home && chmod 1777 /opt/agent-home

CMD ["bash"]

FROM solve-base AS all

ARG CLAUDE_CODE_VERSION
ARG OPENCODE_VERSION

RUN npm install -g \
      "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
      "opencode-ai@${OPENCODE_VERSION}" \
    && claude --version \
    && opencode --version

ENV HOME=/opt/agent-home
RUN mkdir -p /opt/agent-home && chmod 1777 /opt/agent-home

CMD ["bash"]
