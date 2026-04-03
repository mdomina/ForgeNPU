FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/ForgeNPU

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        git \
        iverilog \
        python3 \
        python3-pip \
        python3-venv \
        verilator \
        yosys \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/ForgeNPU

RUN python3 -m pip install --break-system-packages -e .

CMD ["/bin/bash"]
