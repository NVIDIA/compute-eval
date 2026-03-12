# =============================================================================
# Dockerfile.cpp — Sandbox image for C++/CUDA evaluation
# =============================================================================
#
# PURPOSE:
#   This image is the isolated execution sandbox used by DockerEvaluator when
#   grading C++ / CUDA (cuda_cpp) solutions. It provides a complete CUDA
#   development environment with cuDNN and the cudnn-frontend header library,
#   but no Python runtime. One container is spun up per solution, runs
#   build/test/benchmark commands, then is torn down.
#
# IMAGE NAME CONVENTION:
#   compute-eval-cpp:<CUDA_VERSION>   e.g. compute-eval-cpp:13.1.0
#   (pulled from the registry pointed to by $IMAGE_REGISTRY, or nvcr.io/<org>
#    by default — see compute_eval/utils/eval_utils.py:compute_cuda_image)
#
# BUILD :
#   # Default: CUDA 13.1.0
#   docker build -f docker/Dockerfile.cpp \
#                -t compute-eval-cpp:13.1.0 .
#
#   # Override CUDA version:
#   docker build -f docker/Dockerfile.cpp \
#                --build-arg CUDA_VERSION=12.8.0 \
#                --build-arg CUDNN_VERSION=9.20.0.48-1 \
#                -t compute-eval-cpp:12.8.0 .
#
# INCLUDED PACKAGES:
#   - cuDNN runtime + dev libraries (pinned via CUDNN_VERSION build arg)
#   - cudnn-frontend (header-only, cloned from GitHub at build time)
#   - CUPTI-based profiling library (built from compute_eval/cuda_tools/)
#   - Full CUDA development toolchain (nvcc, etc.) from the base image
#
# =============================================================================

ARG CUDA_VERSION=13.1.0
ARG CUDNN_VERSION=9.20.0.48-1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG CUDA_VERSION
ARG CUDNN_VERSION

# Install pinned cuDNN backend libraries
RUN CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    echo "CUDA Version: ${CUDA_VERSION}, cuDNN: ${CUDNN_VERSION}" && \
    apt-get update && \
    apt-get install -y \
        libcudnn9-cuda-${CUDA_MAJOR}=${CUDNN_VERSION} \
        libcudnn9-dev-cuda-${CUDA_MAJOR}=${CUDNN_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# Install git for cudnn-frontend
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install cudnn-frontend headers (header-only library)
RUN git clone --depth 1 --branch main \
        https://github.com/NVIDIA/cudnn-frontend.git /tmp/cudnn-frontend && \
    cp -r /tmp/cudnn-frontend/include/* /usr/local/include/ && \
    rm -rf /tmp/cudnn-frontend

# Copy profiler source files and Makefile
COPY compute_eval/cuda_tools/*.h compute_eval/cuda_tools/*.cpp compute_eval/cuda_tools/Makefile /opt/cuda_tools/

# Build the profiling library using the Makefile
# (Makefile auto-detects architecture and CUPTI paths)
RUN cd /opt/cuda_tools && make && \
    echo "Built profiling library with $(ls *.cpp | wc -l) modules"

# Set runtime environment for profiling
ENV CUPTI_LIB=/usr/local/cuda/lib64 \
    LD_LIBRARY_PATH=/opt/cuda_tools:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CUPTI_PROFILER_LIB=/opt/cuda_tools/libcuda_profile.so

WORKDIR /workspace
CMD ["/bin/bash"]
