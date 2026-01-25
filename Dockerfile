# 使用 NVIDIA 官方 CUDA 12.1 基础镜像（Python 3.10）
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 安装 Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 设置工作目录
WORKDIR /opt/code

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/code \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装 PyTorch (CUDA 12.1 版本，从 PyTorch 官方源)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip show torch

# 复制 requirements.txt 并安装其他 Python 依赖（PyTorch 已安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 复制项目文件
COPY main.py .
COPY utils.py .
COPY app.py .
# config.yaml 不再打包进镜像，使用环境变量代替

# 复制 LoRA 权重文件
COPY zara/pytorch_lora_weights.safetensors ./zara/
COPY hoc/pytorch_lora_weights.safetensors ./hoc/
COPY cos/pytorch_lora_weights.safetensors ./cos/
COPY rl/pytorch_lora_weights.safetensors ./rl/
COPY lulu/pytorch_lora_weights.safetensors ./lulu/

# 创建输出目录
RUN mkdir -p /tmp/images

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# 启动 FastAPI 服务
CMD ["python", "app.py"]
