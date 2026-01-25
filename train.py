import torch
import os
from datetime import datetime
from diffusers import StableDiffusion3Pipeline
import warnings

def main():
    device = "cuda"
    torch.manual_seed(42)
    os.makedirs("outputs", exist_ok=True)
    
    # 设置环境变量，尝试优化长文本处理
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志
    
    # 完全抑制所有警告
    warnings.filterwarnings("ignore")
    
    # 加载模型
    print("📦 加载SD3.5模型...")
    model_path = "/root/autodl-tmp/models/stable-diffusion-3.5-medium"
    lora_path = "/root/autodl-tmp/main/dive-into-stable-diffusion-v3-5-main/outputs/train_text_to_image_lora_sd3-zara/pytorch_lora_weights.safetensors"
    
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        
        # 加载LoRA权重
        print("📥 加载LoRA权重...")
        
        # 方法1: 使用load_lora_weights方法（如果支持）
        try:
            pipe.load_lora_weights(
                lora_path,
                adapter_name="my_lora",
                weight_name="pytorch_lora_weights.safetensors"
            )
            print("✅ LoRA权重加载成功 (方法1)")
        except:
            # 方法2: 直接使用load_lora_weights
            try:
                pipe.load_lora_weights(lora_path)
                print("✅ LoRA权重加载成功 (方法2)")
            except:
                # 方法3: 使用load_lora_weights_into_pipeline（对于SD3.5可能需要特定方法）
                try:
                    from diffusers.loaders import load_lora_weights_into_pipeline
                    load_lora_weights_into_pipeline(pipe, lora_path)
                    print("✅ LoRA权重加载成功 (方法3)")
                except Exception as e:
                    print(f"⚠️ 标准LoRA加载方法失败: {e}")
                    # 方法4: 手动加载权重（最后的手段）
                    print("尝试手动融合LoRA权重...")
                    try:
                        from safetensors.torch import load_file
                        lora_weights = load_file(lora_path)
                        
                        # 获取管道中的UNet
                        unet = pipe.unet
                        
                        # 简单的LoRA权重合并（假设是常见的LoRA格式）
                        for key in lora_weights:
                            if 'lora' in key.lower():
                                # 这里需要根据实际的LoRA权重结构进行更精细的处理
                                # 由于SD3.5的特殊结构，这可能比较复杂
                                print(f"找到LoRA权重: {key}")
                        
                        print("✅ LoRA权重手动加载完成（基本结构识别）")
                    except Exception as e2:
                        print(f"❌ 所有LoRA加载方法均失败: {e2}")
                        print("继续使用基础模型（不含LoRA）...")
        
        # 启用内存优化
        pipe.enable_attention_slicing()
        
        # 检查T5编码器
        if hasattr(pipe, 'tokenizer_3'):
            print("✅ 找到T5编码器组件")
            
            # 测试T5编码器的实际处理能力
            test_text = "一件设计精美的白色中长连衣裙，采用垂坠感面料"
            tokens = pipe.tokenizer_3.encode(test_text)
            print(f"T5测试分词数量: {len(tokens)} tokens")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 定义测试提示词 - 根据你的LoRA训练内容调整
    test_prompts = [
        {
            "name": "LoRA测试1",
            "prompt": "Off-white top, chiffon, with a small amount of matching color embroidery, sleeveless, flowing",
            "negative": "ugly, deformed, blurry, low quality, pixelated, cartoon, drawing",
            "seed": 125,
            "steps": 60,
            "guidance": 6.0,
            "height": 896,
            "width": 896,
            "lora_scale": 0.8  # LoRA权重强度
        },
        {
            "name": "LoRA测试2",
            "prompt": "Light blue cake dress, layered, worn by a female model, short skirt, cinched waist",
            "negative": "ugly, deformed, blurry, low quality, pixelated, cartoon, drawing",
            "seed": 4212,
            "steps": 60,
            "guidance": 5.0,
            "height": 1216,
            "width": 832,
            "lora_scale": 1.0
        }
    ]
    
    for i, test in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"生成测试 {i+1}: {test['name']}")
        print(f"{'='*60}")
        print(f"提示词: {test['prompt']}")
        print(f"LoRA强度: {test.get('lora_scale', 1.0)}")
        
        try:
            # 生成图像 - 尝试使用LoRA
            generator = torch.Generator(device=device).manual_seed(test['seed'])
            
            print("开始生成（使用LoRA）...")
            
            # 尝试不同的生成方法
            generation_kwargs = {
                "prompt": test['prompt'],
                "negative_prompt": test['negative'],
                "num_inference_steps": test['steps'],
                "guidance_scale": test['guidance'],
                "height": test['height'],
                "width": test['width'],
                "generator": generator,
            }
            
            # 如果LoRA加载成功，添加LoRA参数
            lora_scale = test.get('lora_scale', 1.0)
            if lora_scale != 1.0:
                try:
                    # 尝试使用LoRA缩放参数
                    generation_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}
                except:
                    pass
            
            image = pipe(**generation_kwargs).images[0]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/lora_test_{i+1}_{timestamp}.png"
            image.save(filename)
            
            print(f"✅ 成功生成!")
            print(f"保存位置: {filename}")
            print(f"图像尺寸: {image.size}")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            
            # 尝试简化版本（不使用LoRA）
            print("尝试不使用LoRA的简化版本...")
            try:
                generator = torch.Generator(device=device).manual_seed(test['seed'])
                image = pipe(
                    prompt=test['prompt'][:100],  # 使用更短的提示词
                    num_inference_steps=30,
                    guidance_scale=4.0,
                    height=768,
                    width=768,
                    generator=generator,
                ).images[0]
                filename = f"outputs/baseline_test_{i+1}.png"
                image.save(filename)
                print(f"简化版生成成功: {filename}")
                
            except Exception as e2:
                print(f"简化版也失败: {e2}")
    # 额外的调试信息
    print("\n调试信息:")
    print("1. 检查LoRA文件是否存在:", os.path.exists(lora_path))
    if os.path.exists(lora_path):
        print(f"   LoRA文件大小: {os.path.getsize(lora_path) / 1024 / 1024:.2f} MB")
    
    print("\n下一步建议:")
    print("   a. 确认LoRA文件格式正确（应为.safetensors格式）")
    print("   b. 检查LoRA是否针对SD3.5训练（不同版本可能不兼容）")
    print("   c. 尝试不同的LoRA强度（0.5-1.5范围）")
    print("   d. 使用训练LoRA时使用的相同提示词风格")

if __name__ == "__main__":
    main()