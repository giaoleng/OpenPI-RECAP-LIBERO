# LIBERO client environment aligned with the root README (Python 3.10).
# This is a Docker build recipe, NOT proof that an external lab environment has been reproduced.
# The exact external LIBERO commit used for experiments must be restored from the researcher's
# private environment notes before building; no unverified commit hash is invented here.
#
# From repo root, after cloning LIBERO into third_party/libero:
#   docker build -f examples/libero/Dockerfile -t openpi-libero-py310 .
# For the historical Python 3.8 / CUDA 11.3 build see Dockerfile.legacy.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates make g++ clang \
    libosmesa6-dev libgl1-mesa-glx libegl1 libglew-dev \
    libglfw3-dev libgles2-mesa-dev libglib2.0-0 \
    libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"
RUN uv venv --python 3.10 "$UV_PROJECT_ENVIRONMENT"

# LIBERO is intentionally NOT tracked in this Git repo; clone it to third_party/libero
# at the tested commit before building the image.
COPY ./third_party/libero/ /opt/third_party/libero/
COPY ./packages/openpi-client/ /opt/openpi-client/

# Install the standalone simulation client and local policy client.
# Use the README's pinned LeRobot source revision (not the old torch1.11/cu113 lock).
# We allow LeRobot to resolve its dependencies here; the README's --no-deps approach
# is for environments that already have the validated dependency stack installed.
RUN uv pip install -e /opt/openpi-client -e /opt/third_party/libero && \
    uv pip install "git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/app:/app/packages/openpi-client/src:/app/third_party/libero"
ENV LIBERO_CONFIG_PATH=/tmp/libero

RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF

# Runtime: mount the repository at /app, including third_party/libero checkout.
CMD ["/bin/bash"]
