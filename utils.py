import json
import os
import uuid
import datetime
from typing import List, Dict, Any, Optional, Tuple
import requests
import yaml


def get_tos_config_from_env() -> Dict[str, Any]:
    """从 config.yaml 读取 TOS 配置

    Returns:
        Dict[str, Any]: TOS 配置字典

    Raises:
        FileNotFoundError: config.yaml 文件不存在时抛出异常
        ValueError: 配置文件格式错误或缺少必需字段时抛出异常

    示例:
        >>> config = get_tos_config_from_env()
        >>> print(config['bucket_name'])
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"读取配置文件失败: {str(e)}")

    # 提取 tos_config 部分
    tos_config = config.get('tos_config', {})

    # 验证必需的配置项
    required_fields = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name']
    missing_fields = [k for k in required_fields if not tos_config.get(k)]

    if missing_fields:
        raise ValueError(
            f"config.yaml 中缺少必需的 TOS 配置项: {missing_fields}\n"
            f"请确保 config.yaml 包含以下字段:\n"
            f"  tos_config:\n"
            f"    access_key_id: \"your_key_id\"\n"
            f"    access_key_secret: \"your_secret\"\n"
            f"    endpoint: \"tos-cn-beijing.volces.com\"\n"
            f"    region: \"cn-beijing\"\n"
            f"    bucket_name: \"your_bucket\"\n"
            f"    expires_minutes: 1440"
        )

    return tos_config


def upload_file_to_tos(
    file_path: str,
    tos_config: Dict[str, Any],
    object_key_prefix: str = "api_files",
    delete_after_upload: bool = False
) -> Optional[str]:
    """上传文件到火山引擎 TOS 并返回签名 URL

    这是通用的 TOS 上传工具函数。

    Args:
        file_path: 本地文件路径
        tos_config: TOS 配置字典:
            {
                "access_key_id": str,
                "access_key_secret": str,
                "endpoint": str,
                "region": str,
                "bucket_name": str,
                "expires_minutes": int (可选，默认: 30)
            }
        object_key_prefix: TOS 对象键前缀 (例如: "api_images", "video_api_images")
        delete_after_upload: 上传成功后是否删除本地文件 (默认: False)

    Returns:
        临时签名 URL，上传失败返回 None

    Raises:
        ImportError: tos 包不可用时抛出异常
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在: {file_path}")
        return None

    try:
        import tos

        # 初始化 TOS 客户端
        client = tos.TosClientV2(
            ak=tos_config['access_key_id'],
            sk=tos_config['access_key_secret'],
            endpoint=tos_config['endpoint'],
            region=tos_config['region']
        )

        # 生成唯一的对象键 (添加随机数避免并发冲突)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = os.path.basename(file_path)
        object_key = f"{object_key_prefix}/{timestamp}_{unique_id}_{filename}"

        # 上传文件
        print(f"正在上传到 TOS: {filename}")
        result = client.put_object_from_file(
            bucket=tos_config['bucket_name'],
            key=object_key,
            file_path=file_path
        )

        if result.status_code == 200:
            # 生成签名 URL
            expires_minutes = tos_config.get('expires_minutes', 30)
            expires_seconds = expires_minutes * 60

            # 使用 HttpMethodType 枚举
            from tos.enum import HttpMethodType

            signed_url_result = client.pre_signed_url(
                http_method=HttpMethodType.Http_Method_Get,
                bucket=tos_config['bucket_name'],
                key=object_key,
                expires=expires_seconds
            )

            print(f"✓ TOS 上传成功，URL 有效期 {expires_minutes} 分钟")

            # 上传成功后删除本地文件
            if delete_after_upload:
                try:
                    os.remove(file_path)
                    print(f"✓ 已删除本地文件: {file_path}")
                except Exception as e:
                    print(f"⚠ 删除本地文件失败: {e}")

            return signed_url_result.signed_url
        else:
            print(f"✗ TOS upload failed: HTTP {result.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return None
    
def get_oss_config_from_env() -> Dict[str, Any]:
    """从环境变量或 config.yaml 读取 OSS 配置

    优先级：环境变量 > config.yaml

    Returns:
        Dict[str, Any]: OSS 配置字典

    Raises:
        ValueError: 配置项缺失时抛出异常
    """
    # 优先从环境变量读取（生产环境）
    access_key = os.getenv("OSS_ACCESS_KEY_ID")
    if access_key:
        print("✅ 从环境变量读取 OSS 配置")
        config = {
            "access_key_id": access_key,
            "access_key_secret": os.getenv("OSS_ACCESS_KEY_SECRET"),
            "endpoint": os.getenv("OSS_ENDPOINT"),
            "bucket_name": os.getenv("OSS_BUCKET"),
            "expires_minutes": int(os.getenv("OSS_EXPIRES_MINUTES", "43200"))
        }
        # 验证必需字段
        required_fields = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name']
        missing_fields = [k for k in required_fields if not config.get(k)]
        if missing_fields:
            raise ValueError(f"环境变量中缺少 OSS 配置项: {missing_fields}")
        return config

    # 降级：从 config.yaml 读取（本地开发）
    print("⚠️  未检测到环境变量，尝试从 config.yaml 读取 OSS 配置")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"读取配置文件失败: {str(e)}")

    # 提取 oss_config 部分
    oss_config = config.get('oss_config', {})

    # 验证必需的配置项
    required_fields = ['access_key_id', 'access_key_secret', 'endpoint', 'bucket_name']
    missing_fields = [k for k in required_fields if not oss_config.get(k)]

    if missing_fields:
        raise ValueError(
            f"config.yaml 中缺少必需的 OSS 配置项: {missing_fields}\n"
            f"请确保 config.yaml 包含以下字段:\n"
            f"  oss_config:\n"
            f"    access_key_id: \"your_key_id\"\n"
            f"    access_key_secret: \"your_secret\"\n"
            f"    endpoint: \"oss-cn-hangzhou.aliyuncs.com\"\n"
            f"    bucket_name: \"your_bucket\"\n"
            f"    expires_minutes: 43200"
        )

    print("✅ 从 config.yaml 读取 OSS 配置成功")
    return oss_config


def upload_file_to_oss(
    file_path: str,
    oss_config: Dict[str, Any],
    object_key_prefix: str = "api_files",
    delete_after_upload: bool = False
) -> Optional[str]:
    """上传文件到阿里云 OSS 并返回签名 URL

    这是通用的 OSS 上传工具函数。

    Args:
        file_path: 本地文件路径
        oss_config: OSS 配置字典:
            {
                "access_key_id": str,
                "access_key_secret": str,
                "endpoint": str,
                "bucket_name": str,
                "expires_minutes": int (可选，默认: 30)
            }
        object_key_prefix: OSS 对象键前缀 (例如: "api_images", "video_api_images")
        delete_after_upload: 上传成功后是否删除本地文件 (默认: False)

    Returns:
        临时签名 URL，上传失败返回 None

    Raises:
        ImportError: oss2 包不可用时抛出异常
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"✗ 文件不存在: {file_path}")
        return None

    try:
        import oss2

        # 初始化 OSS 客户端
        auth = oss2.Auth(
            oss_config['access_key_id'],
            oss_config['access_key_secret']
        )
        bucket = oss2.Bucket(
            auth,
            oss_config['endpoint'],
            oss_config['bucket_name']
        )

        # 生成唯一的对象键 (添加随机数避免并发冲突)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = os.path.basename(file_path)
        object_key = f"{object_key_prefix}/{timestamp}_{unique_id}_{filename}"

        # 上传文件
        print(f"正在上传到 OSS: {filename}")
        result = bucket.put_object_from_file(object_key, file_path, headers={"x-oss-object-acl": "public-read"})

        if result.status == 200:
            # 生成签名 URL
            expires_minutes = oss_config.get('expires_minutes', 30)
            expires_seconds = expires_minutes * 60
            signed_url = bucket.sign_url('GET', object_key, expires_seconds)

            print(f"✓ OSS 上传成功，URL 有效期 {expires_minutes} 分钟")

            # 上传成功后删除本地文件
            if delete_after_upload:
                try:
                    os.remove(file_path)
                    print(f"✓ 已删除本地文件: {file_path}")
                except Exception as e:
                    print(f"⚠ 删除本地文件失败: {e}")

            return signed_url
        else:
            print(f"✗ OSS 上传失败: HTTP {result.status}")
            return None

    except Exception as e:
        print(f"✗ 上传失败: {e}")
        return None

def download_file(url: str, save_path: str) -> bool:
    """
    下载指定 URL 的内容并保存到本地文件。

    Args:
        url (str): 要下载的文件的 URL。
        save_path (str): 本地保存文件的路径。

    Returns:
        bool: 下载成功返回 True，失败返回 False。
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查 HTTP 请求是否成功

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def get_qwen_config_from_env() -> Dict[str, Any]:
    """从环境变量或 config.yaml 读取通义千问配置

    优先级：环境变量 > config.yaml

    Returns:
        Dict[str, Any]: Qwen 配置字典

    Raises:
        ValueError: 配置项缺失时抛出异常
    """
    # 优先从环境变量读取（生产环境）
    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        print("✅ 从环境变量读取通义千问配置")
        return {
            "api_key": api_key,
            "model": os.getenv("QWEN_MODEL", "qwen-turbo-latest")
        }

    # 降级：从 config.yaml 读取（本地开发）
    print("⚠️  未检测到环境变量，尝试从 config.yaml 读取通义千问配置")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"读取配置文件失败: {str(e)}")

    # 提取 qwen_config 部分
    qwen_config = config.get('qwen_config', {})

    # 验证必需的配置项
    if not qwen_config.get('api_key'):
        raise ValueError(
            f"config.yaml 中缺少通义千问 API Key\n"
            f"请确保 config.yaml 包含:\n"
            f"  qwen_config:\n"
            f"    api_key: \"your-qwen-api-key\"\n"
            f"    model: \"qwen-turbo\""
        )

    print("✅ 从 config.yaml 读取通义千问配置成功")
    return qwen_config


def generate_title_qwen(prompt: str) -> Tuple[str, str]:
    """使用通义千问翻译提示词并生成商品标题

    Args:
        prompt: 商品描述（可能是中文或英文）

    Returns:
        (english_prompt, title): tuple
            - english_prompt: 翻译后的英文提示词（用于SD生成）
            - title: 生成的中文标题（15字以内）
    """
    try:
        qwen_config = get_qwen_config_from_env()
    except Exception as e:
        print(f"[警告] 获取通义千问配置失败: {e}")
        # 降级：原文返回 + 简单标题
        return prompt, "新品推荐"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=qwen_config['api_key'],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        system_prompt = """你是一个时尚电商助手。请完成两个任务：
1. 将商品描述翻译成简洁的英文（用于AI图像生成）
2. 生成一个中文商品标题

请严格按照JSON格式返回，不要任何其他文字：
{"english_prompt": "英文描述", "title": "中文标题"}

标题要求：
- 格式为'修饰词+主体'，例如'米白色无袖上衣'、'黑色露肩连衣裙'
- 总字数严格控制在15字以内
- 修饰词包括颜色、版型、材质等
- 主体包括上衣、连衣裙、短袖、夹克等服装类型"""

        print(f"正在调用通义千问处理提示词...")

        completion = client.chat.completions.create(
            model=qwen_config.get('model', 'qwen-turbo-latest'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"商品描述：{prompt}\n\n请返回JSON格式结果："}
            ],
            temperature=0.7,
            max_tokens=200
        )

        # 获取返回内容
        content = completion.choices[0].message.content.strip()

        # 尝试解析 JSON
        try:
            # 清理可能的 markdown 代码块标记
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(content)
            english_prompt = parsed.get("english_prompt", prompt)
            title = parsed.get("title", "新品推荐")

            print(f"[成功] 英文提示词: {english_prompt}")
            print(f"[成功] 中文标题: {title}")
            return english_prompt, title

        except json.JSONDecodeError as e:
            print(f"[警告] JSON 解析失败: {e}")
            print(f"[调试] 原始返回: {content[:200]}...")
            return prompt, "新品推荐"

    except ImportError:
        print(f"[警告] 未安装 openai 库，请运行: pip install openai")
        return prompt, "新品推荐"
    except Exception as e:
        print(f"[警告] 提示词处理失败: {e}")
        return prompt, "新品推荐"
