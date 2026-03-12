FROM ubuntu:24.04

WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    jq \
    curl \
    wget \
    unzip \
    gnupg \
    software-properties-common \
    ca-certificates \
    lsb-release \
    iptables \
    supervisor \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Docker Engine
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit (officially supported on Ubuntu)
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Configure NVIDIA Container Runtime
RUN mkdir -p /etc/docker && \
    cat > /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

# Set NVIDIA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Symlink python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

# Copy dependency files and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source code and install project
COPY . ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Set paths
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

# Create entrypoint
RUN cat > /entrypoint.sh <<'EOF'
#!/bin/bash
set -e

echo "Starting Docker daemon..."

# Start Docker daemon in background
dockerd \
    --host=unix:///var/run/docker.sock \
    > /var/log/docker.log 2>&1 &

DOCKER_PID=$!
echo "Docker daemon PID: $DOCKER_PID"

# Wait for Docker to be ready
echo "Waiting for Docker daemon..."
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "✓ Docker daemon ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Docker daemon failed to start"
        cat /var/log/docker.log
        exit 1
    fi
    sleep 1
done

# Execute the main command
exec "$@"
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
