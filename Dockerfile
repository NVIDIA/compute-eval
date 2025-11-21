FROM  nvidia/cuda:12.3.2-cudnn9-devel-ubi8

# Python 311 install as the default python3.9 is not compatible with compute eval
RUN yum install -y python3.11 && \
    rm -f /usr/bin/python3 && \
    ln  -s /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    rm -rf /var/lib/apt/lists/* && \
    yum clean all && \
    rm -rf /var/cache/yum

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /compute-eval

# Copy dependency files and install dependencies only (no project)
COPY pyproject.toml uv.lock ./
# Note: To enable Python CUDA support, add --extra python-cuda to both RUN commands
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and install project
COPY . ./
ADD ./data  /compute-eval-data
RUN uv sync --frozen --no-dev

ENV PATH="/compute-eval/.venv/bin:$PATH:/compute-eval"

#set entry point
ENTRYPOINT ["compute_eval"]

# To generate samples do this
# docker run -it  --runtime nvidia -v /home/ubuntu/compute-eval/data:/data  -e  NEMO_API_KEY=$APIKEY compute-eval generate_samples <Include your general samples parameters...>


# To verify correctness do this
# docker run -it  --runtime nvidia -v /home/ubuntu/compute-eval/data:/data compute-eval evaluate_functional_correctness <Include your evaluation functional correctness parameters here..>
