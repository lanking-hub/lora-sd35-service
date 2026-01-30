"""
阿里云函数计算 GPU 函数的 Web 服务入口
使用 FastAPI 处理 HTTP 请求
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# 导入主逻辑
from main import main as generate_image_main

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="LoRA Image Generation Service",
    description="基于 Stable Diffusion 3.5 和 LoRA 的图像生成服务",
    version="1.0.0"
)


class ImageRequest(BaseModel):
    """图像生成请求"""
    brand: str = Field(..., description="品牌名称: zara, hoc, cos, rl, lulu")
    prompt: str = Field(..., description="提示词（中文或英文）")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "brand": "zara",
                    "prompt": "Off-white top, chiffon, with a small amount of matching color embroidery, sleeveless, flowing"
                }
            ]
        }


class ImageResponse(BaseModel):
    """图像生成响应"""
    success: bool = Field(..., description="是否成功")
    url: Optional[str] = Field(None, description="生成的图像 URL")
    title: Optional[str] = Field(None, description="生成的标题")
    error: Optional[str] = Field(None, description="错误信息")


@app.get("/", tags=["根路径"])
async def root():
    """根路径，服务信息"""
    return {
        "service": "LoRA Image Generation Service",
        "version": "1.0.0",
        "status": "running",
        "supported_brands": ["zara", "hoc", "cos", "rl", "lulu"],
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "brands": "/brands",
            "gpu": "/gpu"
        }
    }


@app.get("/health", tags=["健康检查"])
async def health():
    """健康检查接口（函数计算需要）"""
    return {"status": "healthy"}


@app.post("/generate", response_model=ImageResponse, tags=["图像生成"])
async def generate(request: ImageRequest):
    """生成图像接口

    Args:
        request: 包含 brand 和 prompt 的请求

    Returns:
        ImageResponse: 包含图像 URL 和标题
    """
    try:
        # 验证品牌
        valid_brands = ["zara", "hoc", "cos", "rl", "lulu"]
        if request.brand not in valid_brands:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid brand '{request.brand}'. Must be one of: {', '.join(valid_brands)}"
            )

        # 验证提示词
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        logger.info(f"收到请求: brand={request.brand}, prompt={request.prompt[:50]}...")

        # 调用主逻辑
        event = {
            "brand": request.brand,
            "prompt": request.prompt
        }

        success, result = generate_image_main(event)

        if success:
            logger.info(f"生成成功: url={result['url'][:50]}..., title={result['title']}")
            return ImageResponse(
                success=True,
                url=result["url"],
                title=result["title"]
            )
        else:
            logger.error(f"生成失败: {result}")
            return ImageResponse(
                success=False,
                error=str(result)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("生成图像时发生未预期的错误")
        return ImageResponse(
            success=False,
            error=f"Unexpected error: {str(e)}"
        )


@app.get("/brands", tags=["品牌列表"])
async def list_brands():
    """获取支持的品牌列表"""
    return {
        "brands": [
            {"name": "zara", "description": "Zara 品牌 LoRA"},
            {"name": "hoc", "description": "HOC 品牌 LoRA"},
            {"name": "cos", "description": "COS 品牌 LoRA"},
            {"name": "rl", "description": "RL 品牌 LoRA"},
            {"name": "lulu", "description": "Lulu 品牌 LoRA"}
        ]
    }


@app.get("/gpu", tags=["GPU信息"])
async def gpu_info():
    """获取GPU信息"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = round(props.total_memory / 1024**3, 2)
                devices.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": total_memory_gb,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                })

            return {
                "cuda_available": True,
                "device_count": device_count,
                "devices": devices,
                "current_device": torch.cuda.current_device(),
                "torch_version": torch.__version__
            }
        else:
            return {
                "cuda_available": False,
                "message": "CUDA not available, using CPU"
            }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to get GPU info"
        }


# 启动服务器
if __name__ == "__main__":
    port = int(os.getenv("PORT", "9000"))
    logger.info(f"启动服务: port={port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
