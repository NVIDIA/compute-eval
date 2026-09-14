#!/bin/bash
set -e

# Generate CDI spec against the actual host GPU devices at runtime.
# This must happen at startup, not build time, because the GPU device
# files are not present during docker build.
mkdir -p /etc/cdi
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

exec "$@"
