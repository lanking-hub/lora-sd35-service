# 使用阿里云函数计算官方 PyTorch 基础镜像
# 包含 PyTorch 和 CUDA 支持，平台会在运行时注入 GPU 驱动
FROM registry.cn-shanghai.aliyuncs.com/serverless_devs/pytorch:22.12-py3

# 设置工作目录
WORKDIR /opt/code

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/code \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# 复制 requirements.txt 并安装 Python 依赖
# 注意：PyTorch 已包含在基础镜像中，无需单独安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com && \
    pip install --upgrade 'peft>=0.17.0' --index-url https://pypi.org/simple/

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
