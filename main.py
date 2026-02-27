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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置临时文件目录
if os.name == 'nt':
    os.environ['TMP'] = 'D:\\temp'
    os.environ['TEMP'] = 'D:\\temp'
else:
    os.environ['TMP'] = '/tmp'
    os.environ['TEMP'] = '/tmp'

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# OSS 配置
OSS_MOUNT_POINT = os.getenv("OSS_MOUNT_POINT", "/mnt/oss")
OSS_MODEL_PATH = os.path.join(OSS_MOUNT_POINT, "models", "sd35-medium")

# 基础模型路径
if os.getenv("BASE_MODEL_PATH"):
    BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
    print(f"✅ 使用环境变量指定的模型路径: {BASE_MODEL_PATH}")
elif os.path.exists(OSS_MODEL_PATH):
    BASE_MODEL_PATH = OSS_MODEL_PATH
    print(f"✅ 检测到 OSS 挂载，使用 OSS 模型路径: {BASE_MODEL_PATH}")
else:
    BASE_MODEL_PATH = "stabilityai/stable-diffusion-3.5-medium"
    print(f"⚠️  未检测到 OSS 挂载，将使用 HuggingFace Hub: {BASE_MODEL_PATH}")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 品牌 LoRA 映射
BRAND_LORA_MAP = {
    "zara": os.path.join(PROJECT_ROOT, "zara", "pytorch_lora_weights.safetensors"),
    "hoc": os.path.join(PROJECT_ROOT, "hoc", "pytorch_lora_weights.safetensors"),
    "cos": os.path.join(PROJECT_ROOT, "cos", "pytorch_lora_weights.safetensors"),
    "rl": os.path.join(PROJECT_ROOT, "rl", "pytorch_lora_weights.safetensors"),
    "lulu": os.path.join(PROJECT_ROOT, "lulu", "pytorch_lora_weights.safetensors"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_pipe: Any = None


def load_pipeline():
    """延迟加载 pipeline"""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"📦 正在加载 SD3.5 模型...")
    print(f"   模型路径: {BASE_MODEL_PATH}")
    print(f"   设备: {DEVICE}")

    try:
        print("   正在加载模型组件（这可能需要几分钟）...")

        if DEVICE == "cuda":
            print("   ✅ 使用 GPU 模式（快速）")
            dtype = torch.float16
        else:
            print("   ⚠️  使用 CPU 模式（较慢，约 30 分钟/张）")
            dtype = torch.float32

        is_local_path = (
            os.path.exists(BASE_MODEL_PATH) or
            BASE_MODEL_PATH.startswith("/") or
            BASE_MODEL_PATH.startswith("./") or
            BASE_MODEL_PATH.startswith("../") or
            (len(BASE_MODEL_PATH) > 1 and BASE_MODEL_PATH[1] == ':')
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if is_local_path:
            load_kwargs["local_files_only"] = True
            print(f"   🔒 使用本地文件模式 (local_files_only=True)")

        _pipe = StableDiffusion3Pipeline.from_pretrained(
            BASE_MODEL_PATH,
            **load_kwargs
        )

        # 直接加载到GPU
        if DEVICE == "cuda":
            print("   直接加载到GPU（不使用CPU offload）  ")
            _pipe = _pipe.to(DEVICE)
        else:
            _pipe = _pipe.to(DEVICE)

        _pipe.enable_attention_slicing()
        print("✅ 模型加载成功")

    except Exception as e:
        import traceback
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        raise RuntimeError(f"模型加载失败: {str(e)}")

    return _pipe


def validate_request(event: Dict[str, Any]) -> Tuple[str, str]:
    """验证请求数据"""
    brand = event.get("brand", "").lower()
    prompt = event.get("prompt", "")

    if not brand:
        raise ValueError("缺少 'brand' 参数")

    if brand not in BRAND_LORA_MAP:
        raise ValueError(f"不支持的品牌: {brand}，支持的品牌: {', '.join(BRAND_LORA_MAP.keys())}")

    if not prompt:
        raise ValueError("缺少 'prompt' 参数")

    return brand, prompt


def translate_prompt(prompt: str) -> str:
    """翻译中文提示词到英文"""
    try:
        from openai import OpenAI
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            print("⚠️  未配置 QWEN_API_KEY，跳过翻译")
            return prompt

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        print(f"🌐 正在翻译提示词...")
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的翻译，将中文翻译成适合 AI 绘画的英文提示词。只返回翻译结果，不要解释。"},
                {"role": "user", "content": prompt}
            ]
        )

        translated = completion.choices[0].message.content.strip()
        print(f"   翻译结果: {translated}")
        return translated

    except Exception as e:
        print(f"⚠️  翻译失败: {e}，使用原提示词")
        return prompt


def generate_title(english_prompt: str) -> str:
    """生成中文产品标题"""
    try:
        result = generate_title_qwen(english_prompt)
        # generate_title_qwen 返回元组 (英文提示词, 中文标题)
        # 需要提取第二个元素（中文标题）
        if isinstance(result, tuple) and len(result) > 1:
            title = result[1]
        else:
            title = result
        
        if len(title) > 15:
            title = title[:15]
        return title
    except Exception as e:
        print(f"⚠️  标题生成失败: {e}")
        return "AI生成图像"


def generate_image(brand: str, prompt: str) -> str:
    """生成图像"""
    print(f"\n{'='*60}")
    print(f"🎨 开始生成图像")
    print(f"{'='*60}")

    try:
        pipe = load_pipeline()

        lora_path = BRAND_LORA_MAP[brand]
        print(f"📌 LoRA 权重: {lora_path}")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 权重文件不存在: {lora_path}")

        # 🔧 智能LoRA管理：检查adapter是否已加载
        print(f"🔍 检查 LoRA adapter 状态...")
        
        # 获取当前已加载的所有adapters
        if hasattr(pipe, 'get_active_adapters'):
            active_adapters = pipe.get_active_adapters()
        else:
            active_adapters = []
        
        print(f"   当前激活的adapters: {active_adapters}")
        
        # 判断是否需要加载新adapter
        if brand in active_adapters:
            # 已加载，直接激活使用
            print(f"✅ LoRA '{brand}' 已加载，直接使用")
            pipe.set_adapters([brand])
        else:
            # 尝试加载新adapter
            try:
                pipe.load_lora_weights(lora_path, adapter_name=brand)
                pipe.set_adapters([brand])
                print(f"✅ 成功加载 LoRA: {brand}")
            except ValueError as e:
                if "already in use" in str(e):
                    # 已存在但未激活，直接激活
                    print(f"✅ LoRA '{brand}' 已存在，激活使用")
                    pipe.set_adapters([brand])
                else:
                    raise e

        combined_prompt = f"fashion clothing, {prompt}, garment, apparel, high quality, professional fashion photography"

        print(f"📝 提示词: {combined_prompt}")
        print(f"⏳ 开始生成...（这可能需要 20-40 秒）")

        with torch.no_grad():
            result = pipe(
                prompt=combined_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=1024,
                width=1024,
            )

        os.makedirs("/tmp/images", exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join("/tmp/images", filename)
        result.images[0].save(image_path)

        print(f"✅ 图像已保存: {image_path}")
        return image_path

    except Exception as e:
        import traceback
        print(f"❌ 图像生成失败: {e}")
        traceback.print_exc()
        raise RuntimeError(f"图像生成失败: {str(e)}")


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """函数计算入口"""
    try:
        print(f"\n{'='*60}")
        print(f"收到新的图像生成请求")
        print(f"{'='*60}")
        print(f"📋 请求参数:")
        print(f"   品牌: {event.get('brand', '')}")
        print(f"   提示词长度: {len(event.get('prompt', ''))} 字符")

        # 验证请求
        brand, prompt = validate_request(event)

        print(f"\n🔍 LoRA 配置:")
        print(f"   路径: {BRAND_LORA_MAP[brand]}")
        print(f"   文件存在: {os.path.exists(BRAND_LORA_MAP[brand])}")

        # 一次调用完成翻译和标题生成（拼接服装上下文避免歧义词被误判）
        english_prompt, title = generate_title_qwen(f"服装款式/面料描述，生成服装图像：{prompt}")
        print(f"🌐 翻译结果: {english_prompt}")
        print(f"📝 标题: {title}")

        # 生成图像
        image_path = generate_image(brand, english_prompt)

        # 上传到 OSS
        print(f"\n☁️  上传到 OSS...")
        oss_config = get_oss_config_from_env()
        image_url = upload_file_to_oss(image_path, oss_config)

        print(f"\n✅ 处理完成!")
        print(f"   图片 URL: {image_url}")

        return {
            "success": True,
            "image_url": image_url,
            "title": title
        }

    except ValueError as e:
        # 参数验证错误
        print(f"\n❌ 参数错误: {e}")
        return {
            "success": False,
            "url": None,
            "title": None,
            "error": str(e)
        }

    except Exception as e:
        # 其他错误
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "url": None,
            "title": None,
            "error": str(e)
        }


# ========== FastAPI 兼容层 ==========
def main(event, context=None):
    """FastAPI app.py 兼容的入口函数
    
    Args:
        event: 事件字典，包含 brand 和 prompt
        context: 上下文（可选）
    
    Returns:
        (success, result): 元组格式，兼容 app.py 调用
    """
    result = handler(event, context)
    
    if result.get("success"):
        return True, {
            "url": result["image_url"],
            "title": result["title"]
        }
    else:
        return False, result.get("error", "Unknown error")


if __name__ == "__main__":
    # 本地测试
    import json
    test_event = {
        "brand": "zara",
        "prompt": "一件优雅的红色晚礼服，适合正式场合"
    }
    result = handler(test_event)
    print(f"\n测试结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
