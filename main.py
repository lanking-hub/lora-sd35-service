import os
import torch
import uuid
from datetime import datetime
from diffusers import StableDiffusion3Pipeline
import warnings
from typing import Tuple, Dict, Any
from utils import upload_file_to_oss, get_oss_config_from_env, generate_title_qwen

# 抑制警告
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 禁用 tokenizer 并行转换，避免卡住
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# 设置临时文件目录（跨平台兼容）
if os.name == 'nt':  # Windows
    os.environ['TMP'] = 'D:\\temp'
    os.environ['TEMP'] = 'D:\\temp'
else:  # Linux/Docker
    os.environ['TMP'] = '/tmp'
    os.environ['TEMP'] = '/tmp'

# ============= 全局配置 =============
# 设置 Hugging Face 镜像（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# OSS 挂载配置（函数计算部署时使用）
OSS_MOUNT_POINT = os.getenv("OSS_MOUNT_POINT", "/mnt/oss")  # OSS 挂载点
OSS_MODEL_PATH = os.path.join(OSS_MOUNT_POINT, "models", "sd35-medium")  # OSS 上的模型路径

# 基础模型路径配置
# 优先级：环境变量 > OSS 挂载路径 > Hugging Face Hub
# 注意：不再使用 /tmp 缓存，因为 44GB 模型超过临时空间限制（10GB）
if os.getenv("BASE_MODEL_PATH"):
    # 部署时通过环境变量指定（优先级最高）
    BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
    print(f"✅ 使用环境变量指定的模型路径: {BASE_MODEL_PATH}")
elif os.path.exists(OSS_MODEL_PATH):
    # 函数计算环境：直接使用 OSS 挂载路径
    BASE_MODEL_PATH = OSS_MODEL_PATH
    print(f"✅ 检测到 OSS 挂载，使用 OSS 模型路径: {BASE_MODEL_PATH}")
else:
    # 本地开发：使用 Hugging Face Hub
    BASE_MODEL_PATH = "stabilityai/stable-diffusion-3.5-medium"
    print(f"⚠️  未检测到 OSS 挂载，将使用 HuggingFace Hub: {BASE_MODEL_PATH}")

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 品牌到 LoRA 文件的映射
BRAND_LORA_MAP = {
    "zara": os.path.join(PROJECT_ROOT, "zara", "pytorch_lora_weights.safetensors"),
    "hoc": os.path.join(PROJECT_ROOT, "hoc", "pytorch_lora_weights.safetensors"),
    "cos": os.path.join(PROJECT_ROOT, "cos", "pytorch_lora_weights.safetensors"),
    "rl": os.path.join(PROJECT_ROOT, "rl", "pytorch_lora_weights.safetensors"),
    "lulu": os.path.join(PROJECT_ROOT, "lulu", "pytorch_lora_weights.safetensors"),
}

# 设备配置 - 自动选择 GPU 或 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局 pipeline 对象 (复用以减少加载时间)
_pipe: Any = None


def load_pipeline():
    """延迟加载 pipeline,只在第一次调用时加载

    Returns:
        StableDiffusion3Pipeline: 加载好的 pipeline
    """
    global _pipe

    if _pipe is not None:
        return _pipe

    print(f"📦 正在加载 SD3.5 模型...")
    print(f"   模型路径: {BASE_MODEL_PATH}")
    print(f"   设备: {DEVICE}")

    try:
        print("   正在加载模型组件（这可能需要几分钟）...")

        # 根据设备选择数据类型
        if DEVICE == "cuda":
            print("   ✅ 使用 GPU 模式（快速）")
            dtype = torch.float16
        else:
            print("   ⚠️  使用 CPU 模式（较慢，约 30 分钟/张）")
            dtype = torch.float32

        # 判断是否从本地路径加载（OSS 挂载或环境变量指定）
        is_local_path = (
            os.path.exists(BASE_MODEL_PATH) or  # 路径存在
            BASE_MODEL_PATH.startswith("/") or  # Linux 绝对路径
            BASE_MODEL_PATH.startswith("./") or  # 相对路径
            BASE_MODEL_PATH.startswith("../") or
            (len(BASE_MODEL_PATH) > 1 and BASE_MODEL_PATH[1] == ':')  # Windows 路径 (C:\, E:\, ...)
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        # 如果是本地路径，添加 local_files_only=True 避免访问 HuggingFace
        if is_local_path:
            load_kwargs["local_files_only"] = True
            print(f"   🔒 使用本地文件模式 (local_files_only=True)")

        _pipe = StableDiffusion3Pipeline.from_pretrained(
            BASE_MODEL_PATH,
            **load_kwargs
        ).to(DEVICE)

        # 启用内存优化
        _pipe.enable_attention_slicing()

        print("✅ 模型加载成功")

    except Exception as e:
        import traceback
        print(f"❌ 模型加载失败: {e}")
        print(f"\n详细错误信息:")
        traceback.print_exc()
        raise RuntimeError(f"模型加载失败: {str(e)}")

    return _pipe


def validate_request(event: Dict[str, Any]) -> Tuple[str, str]:
    """验证请求数据

    Args:
        event: 请求数据字典

    Returns:
        (brand, prompt): 品牌名称和提示词

    Raises:
        ValueError: 请求数据无效时抛出异常
    """
    brand = event.get("brand", "").lower()
    prompt = event.get("prompt", "")

    # 验证品牌
    if not brand:
        raise ValueError("缺少 'brand' 参数")

    if brand not in BRAND_LORA_MAP:
        supported_brands = list(BRAND_LORA_MAP.keys())
        raise ValueError(
            f"不支持的品牌: '{brand}'。"
            f"支持的品牌: {supported_brands}"
        )

    # 验证 prompt
    if not prompt:
        raise ValueError("缺少 'prompt' 参数")

    if not isinstance(prompt, str):
        raise ValueError("'prompt' 必须是字符串类型")

    if len(prompt) > 2000:
        raise ValueError("'prompt' 长度不能超过 2000 字符")

    return brand, prompt


def load_lora_weights(pipe, lora_path: str) -> None:
    """加载 LoRA 权重

    Args:
        pipe: Stable Diffusion pipeline
        lora_path: LoRA 权重文件路径

    Raises:
        RuntimeError: LoRA 加载失败时抛出异常
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA 文件不存在: {lora_path}")

    print(f"📥 正在加载 LoRA 权重...")
    print(f"   LoRA 路径: {lora_path}")

    try:
        # 尝试多种加载方法
        try:
            pipe.load_lora_weights(
                lora_path,
                adapter_name="brand_lora",
                weight_name="pytorch_lora_weights.safetensors"
            )
            print("✅ LoRA 权重加载成功 (方法1)")
        except Exception as e1:
            print(f"   方法1 失败: {e1}")
            try:
                pipe.load_lora_weights(lora_path)
                print("✅ LoRA 权重加载成功 (方法2)")
            except Exception as e2:
                print(f"   方法2 失败: {e2}")
                raise RuntimeError(
                    f"LoRA 权重加载失败。请检查文件格式是否正确。"
                )

    except Exception as e:
        raise RuntimeError(f"LoRA 权重加载失败: {str(e)}")


def generate_image(pipe, prompt: str, seed: int = 42) -> Any:
    """生成图像

    Args:
        pipe: Stable Diffusion pipeline
        prompt: 提示词
        seed: 随机种子

    Returns:
        生成的 PIL Image 对象

    Raises:
        RuntimeError: 图像生成失败时抛出异常
    """
    print(f"🎨 正在生成图像...")
    print(f"   提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    try:
        generator = torch.Generator(DEVICE).manual_seed(seed)

        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=6.0,
            height=896,
            width=896,
            generator=generator,
        ).images[0]

        print(f"✅ 图像生成成功 (尺寸: {image.size})")
        return image

    except Exception as e:
        raise RuntimeError(f"图像生成失败: {str(e)}")


def save_and_upload_image(image, brand: str) -> str:
    """保存图像到本地并上传到阿里云 OSS

    Args:
        image: PIL Image 对象
        brand: 品牌名称

    Returns:
        str: OSS 签名 URL

    Raises:
        RuntimeError: 保存或上传失败时抛出异常
    """
    # 创建输出目录（函数计算使用 /tmp，本地开发使用当前目录）
    if os.path.exists("/tmp"):  # 函数计算环境
        output_dir = "/tmp/images"
    else:  # 本地开发环境
        output_dir = os.path.join(PROJECT_ROOT, "lora_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{brand}_{timestamp}_{unique_id}.png"
    file_path = os.path.join(output_dir, filename)

    # 保存到本地
    print(f"💾 正在保存图像...")
    try:
        image.save(file_path)
        print(f"✅ 图像已保存: {file_path}")
    except Exception as e:
        raise RuntimeError(f"图像保存失败: {str(e)}")

    # 上传到 OSS
    print(f"☁️  正在上传到阿里云 OSS...")
    try:
        oss_config = get_oss_config_from_env()
        image_url = upload_file_to_oss(
            file_path=file_path,
            oss_config=oss_config,
            object_key_prefix="lora_images",
            delete_after_upload=True  # 上传后删除本地文件
        )

        if not image_url:
            raise RuntimeError("OSS 上传返回空 URL")

        print(f"✅ 上传成功: {image_url}")
        return image_url

    except Exception as e:
        # 上传失败时保留本地文件
        print(f"⚠️  OSS 上传失败: {e}")
        print(f"   本地文件保留: {file_path}")
        raise RuntimeError(f"图像上传 OSS 失败: {str(e)}")


def main(event: Dict[str, Any]) -> Tuple[bool, Any]:
    """主函数:接收 brand 和 prompt,生成图像并返回 URL 和标题

    Args:
        event: 请求数据字典,包含:
            - brand: 品牌名称 (zara/hoc/cos/rl/lulu)
            - prompt: 图像生成提示词

    Returns:
        (success, result): tuple
            - success: bool, 是否成功
            - result: 成功时为 dict {"url": str, "title": str}
                     失败时为 str (错误信息)

    示例:
        >>> event = {"brand": "zara", "prompt": "A white dress"}
        >>> success, result = main(event)
        >>> if success:
        ...     print(result["url"], result["title"])
    """
    print(f"\n{'='*60}")
    print(f"收到新的图像生成请求")
    print(f"{'='*60}\n")

    try:
        # 1. 验证请求数据
        brand, prompt = validate_request(event)
        print(f"📋 请求参数:")
        print(f"   品牌: {brand}")
        print(f"   提示词长度: {len(prompt)} 字符\n")

        # 2. 获取 LoRA 路径
        lora_path = BRAND_LORA_MAP[brand]
        print(f"🔍 LoRA 配置:")
        print(f"   路径: {lora_path}")
        print(f"   文件存在: {os.path.exists(lora_path)}\n")

        # 3. 加载 pipeline
        pipe = load_pipeline()

        # 4. 加载 LoRA 权重
        load_lora_weights(pipe, lora_path)

        # 5. 处理提示词（翻译+生成标题）
        print(f"\n正在处理提示词...")
        english_prompt, title = generate_title_qwen(prompt)
        print(f"   原文: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"   英文: {english_prompt}")
        print(f"   标题: {title}\n")

        # 6. 生成图像（使用翻译后的英文提示词）
        image = generate_image(pipe, english_prompt, seed=42)

        # 6. 保存并上传
        image_url = save_and_upload_image(image, brand)

        print(f"\n{'='*60}")
        print(f"✅ 图像生成完成")
        print(f"{'='*60}\n")

        return True, {"url": image_url, "title": title}

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 处理失败: {str(e)}")
        print(f"{'='*60}\n")
        return False, str(e)


# ============= 本地测试 =============
if __name__ == "__main__":
    # 测试事件
    test_event = {
        "brand": "zara",
        "prompt": "Off-white top, chiffon, with a small amount of matching color embroidery, sleeveless, flowing"
    }

    # 运行测试
    success, result = main(test_event)

    if success:
        print(f"\n✅ 测试成功!")
        print(f"图像 URL: {result['url']}")
        print(f"标题: {result['title']}")
    else:
        print(f"\n❌ 测试失败!")
        print(f"错误信息: {result}")
