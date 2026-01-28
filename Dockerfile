# 使用官方Python 3.10镜像（通过docker.1ms.run镜像加速器拉取）
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

# 安装PyTorch 2.4.1 CUDA版本（支持CUDA 12.1，包含torch.library.custom_op以兼容diffusers 0.35.0）
# PyTorch 2.4.1+ 才有 torch.library.custom_op，2.4.0 不支持
# PyTorch 2.6.0 不支持 CUDA 12.1（只支持 11.8, 12.4, 12.6），函数计算环境是 CUDA 12.1
# 明确指定+cu121后缀确保下载CUDA版本，避免下载CPU版本
# 增加timeout到1200秒避免下载大文件超时
RUN pip install --no-cache-dir --timeout 1200 \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 降级numpy到1.x以兼容PyTorch
RUN pip install --no-cache-dir --timeout 1200 "numpy<2.0"

# 安装其他Python依赖（先从阿里云镜像安装基础依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 1200 -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 从官方PyPI安装Hugging Face库并锁定版本
RUN pip install --no-cache-dir --timeout 1200 \
    'diffusers==0.35.0' \
    'transformers==4.46.0' \
    'accelerate==1.0.0' \
    'safetensors==0.4.5' \
    'huggingface-hub==0.34.0' \
    'sentencepiece==0.2.0' \
    'peft==0.17.1' \
    --index-url https://pypi.org/simple/ \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# 创建非root用户以满足函数计算安全规范
RUN groupadd -r appuser && useradd -r -g appuser appuser

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
    chown -R appuser:appuser /opt/code && \
    chown -R appuser:appuser /tmp/images && \
    chmod -R 755 /opt/code && \
    chmod -R 777 /tmp/images

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 9000

# 启动 FastAPI 服务（使用绝对路径提升稳定性）
CMD ["python", "/opt/code/app.py"]
