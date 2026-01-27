# 使用纯Python 3.10镜像（非slim版本，包含完整系统库以支持PyTorch CUDA运行时）
# 注意：此镜像不包含CUDA驱动，避免与函数计算运行时注入的驱动冲突
FROM python:3.10

# 设置工作目录
WORKDIR /opt/code

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/code \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（包含编译工具，以防某些包需要编译）
RUN apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装PyTorch 2.1.2（支持CUDA 12.1，但不包含驱动）
# 锁定版本以避免API变更导致的问题
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装其他Python依赖（先从阿里云镜像安装基础依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 从官方PyPI安装Hugging Face库并锁定版本
RUN pip install --no-cache-dir \
    'diffusers==0.36.0' \
    'transformers==4.46.0' \
    'accelerate==1.0.0' \
    'safetensors==0.4.5' \
    'huggingface-hub==0.26.0' \
    'sentencepiece==0.2.0' \
    'peft==0.17.1' \
    --index-url https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

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

# 创建输出目录并设置权限
RUN mkdir -p /tmp/images && \
    chmod -R 755 /opt/code && \
    chmod -R 777 /tmp/images

# 暴露端口
EXPOSE 9000

# 启动 FastAPI 服务
CMD ["python", "app.py"]
