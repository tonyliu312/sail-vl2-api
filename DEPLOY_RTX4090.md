# SAIL-VL2-8B-Thinking RTX 4090 部署指南

## 硬件要求

- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **系统**: Windows 11
- **内存**: 建议 32GB+ RAM
- **存储**: 约 20GB 空间（模型文件）

## 性能预期

| 配置 | 预期性能 |
|------|---------|
| FP16 (无量化) | 30-50 tokens/s |
| INT8 量化 | 50-80 tokens/s |
| INT4 量化 (AWQ/GPTQ) | 80-120 tokens/s |
| Flash Attention 2 + INT4 | 100-150 tokens/s |

> RTX 4090 拥有 Ada Lovelace 架构，支持 FP8/INT8 Tensor Cores，量化后性能提升显著。

---

## 一、环境准备

### 1.1 安装 CUDA Toolkit

```powershell
# 下载 CUDA 12.1+ (推荐 12.4)
# https://developer.nvidia.com/cuda-downloads

# 验证安装
nvidia-smi
nvcc --version
```

### 1.2 安装 Python 环境

```powershell
# 推荐使用 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

conda create -n sailvl python=3.10 -y
conda activate sailvl
```

### 1.3 安装 PyTorch (CUDA 版本)

```powershell
# PyTorch 2.2+ with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 二、安装依赖

### 2.1 基础依赖

```powershell
pip install transformers>=4.40.0
pip install accelerate>=0.27.0
pip install fastapi uvicorn httpx pillow
pip install sentencepiece tiktoken
```

### 2.2 Flash Attention 2 (强烈推荐)

Flash Attention 2 可以提升 30-50% 的推理速度并减少显存占用。

```powershell
# Windows 上安装 Flash Attention 2
pip install flash-attn --no-build-isolation

# 如果上述失败，尝试预编译版本
pip install flash-attn -f https://github.com/Dao-AILab/flash-attention/releases
```

### 2.3 量化支持 (可选但推荐)

```powershell
# bitsandbytes (INT8/INT4 量化)
pip install bitsandbytes>=0.42.0

# AutoGPTQ (GPTQ INT4 量化)
pip install auto-gptq>=0.7.0 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

# AutoAWQ (AWQ INT4 量化)
pip install autoawq>=0.2.0
```

---

## 三、模型下载

```powershell
# 设置 HuggingFace 镜像 (国内用户)
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载模型
huggingface-cli download BytedanceDouyinContent/SAIL-VL2-8B-Thinking --local-dir ./models/SAIL-VL2-8B-Thinking
```

---

## 四、优化配置

### 4.1 创建优化版 model_loader_cuda.py

```python
"""
SAIL-VL2-8B-Thinking CUDA 优化加载器 (RTX 4090)
"""
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
from PIL import Image
import io
import base64
from typing import Optional, List, Union, AsyncGenerator
import httpx
import asyncio
from threading import Thread


class SAILVLModelCUDA:
    def __init__(
        self,
        model_path: str = "BytedanceDouyinContent/SAIL-VL2-8B-Thinking",
        quantization: str = "none",  # "none", "int8", "int4"
        use_flash_attention: bool = True,
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.quantization = quantization
        self.use_flash_attention = use_flash_attention

    def load(self):
        """加载模型 - CUDA 优化版"""
        print(f"正在加载模型: {self.model_path}")
        print(f"量化模式: {self.quantization}")
        print(f"Flash Attention: {self.use_flash_attention}")

        # 加载 tokenizer 和 processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # 配置量化
        quantization_config = None
        torch_dtype = torch.bfloat16  # RTX 4090 原生支持 bfloat16

        if self.quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            print("使用 INT8 量化")

        elif self.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",  # NormalFloat4
            )
            print("使用 INT4 NF4 量化")

        # 配置 Attention 实现
        attn_impl = "flash_attention_2" if self.use_flash_attention else "sdpa"

        # 检查 Flash Attention 是否可用
        if self.use_flash_attention:
            try:
                import flash_attn
                print(f"Flash Attention 版本: {flash_attn.__version__}")
            except ImportError:
                print("Flash Attention 未安装，回退到 SDPA")
                attn_impl = "sdpa"

        print(f"Attention 实现: {attn_impl}")

        # 加载模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            attn_implementation=attn_impl,
            device_map="auto",  # 自动分配到 GPU
            low_cpu_mem_usage=True,
        )

        self.model.eval()

        # 打印显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"显存使用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已预留)")

        print("模型加载完成")

    async def load_image_from_url(self, url: str) -> Image.Image:
        """从URL加载图片"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

    def load_image_from_base64(self, base64_str: str) -> Image.Image:
        """从base64加载图片"""
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    async def process_content(self, content: Union[str, List]) -> tuple[str, Optional[Image.Image]]:
        """处理OpenAI格式的content"""
        if isinstance(content, str):
            return content, None

        text_parts = []
        image = None

        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url", "")

                if url.startswith("data:"):
                    image = self.load_image_from_base64(url)
                else:
                    image = await self.load_image_from_url(url)

        return " ".join(text_parts), image

    async def generate(
        self,
        messages: List[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = True,
    ) -> str:
        """生成响应"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load() 方法")

        # 思考模式提示词
        cot_prompt = ""
        if enable_thinking:
            cot_prompt = r" You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

        # 转换消息格式
        processed_messages = []
        image = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            text, msg_image = await self.process_content(content)
            if msg_image is not None:
                image = msg_image

            if role == "user" and msg == messages[-1] and enable_thinking:
                text = text + cot_prompt

            if image is not None and role == "user":
                processed_messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": "image_placeholder"},
                        {"type": "text", "text": text}
                    ]
                })
            else:
                processed_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": text}]
                })

        # 应用聊天模板
        text = self.processor.apply_chat_template(
            processed_messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # 处理输入
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        # 生成配置
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "use_cache": True,  # CUDA 上启用 KV Cache
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # 生成响应
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 解码响应
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 清理响应
        if '<|im_end|>' in response:
            response = response.split('<|im_end|>')[0].strip()

        if '<|im_start|>assistant' in response:
            response = response.split('<|im_start|>assistant')[-1].strip()

        # 处理思考模式的输出
        thinking_content = None
        final_content = response

        if '<think>' in response:
            import re
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            else:
                parts = response.split('<think>')
                if len(parts) > 1:
                    thinking_content = parts[1].strip()
                    final_content = ""

        if enable_thinking and thinking_content:
            if final_content:
                return f"<think>\n{thinking_content}\n</think>\n\n{final_content}"
            else:
                return f"<think>\n{thinking_content}\n</think>"

        if not final_content and thinking_content:
            return thinking_content

        return final_content if final_content else response

    async def generate_stream(
        self,
        messages: List[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load() 方法")

        cot_prompt = ""
        if enable_thinking:
            cot_prompt = r" You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."

        processed_messages = []
        image = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            text, msg_image = await self.process_content(content)
            if msg_image is not None:
                image = msg_image

            if role == "user" and msg == messages[-1] and enable_thinking:
                text = text + cot_prompt

            if image is not None and role == "user":
                processed_messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": "image_placeholder"},
                        {"type": "text", "text": text}
                    ]
                })
            else:
                processed_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": text}]
                })

        text = self.processor.apply_chat_template(
            processed_messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0
        )

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        def generate_in_thread():
            with torch.inference_mode():
                self.model.generate(**inputs, **gen_kwargs)

        thread = Thread(target=generate_in_thread)
        thread.start()

        import queue
        token_queue = queue.Queue()
        SENTINEL = object()

        def stream_to_queue():
            try:
                for text in streamer:
                    token_queue.put(text)
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(SENTINEL)

        queue_thread = Thread(target=stream_to_queue)
        queue_thread.start()

        loop = asyncio.get_event_loop()

        while True:
            try:
                item = await loop.run_in_executor(
                    None,
                    lambda: token_queue.get(timeout=120.0)
                )

                if item is SENTINEL:
                    break
                elif isinstance(item, Exception):
                    raise item
                elif item:
                    yield item

            except queue.Empty:
                if not thread.is_alive() and not queue_thread.is_alive():
                    break
                continue
            except Exception:
                break

        queue_thread.join(timeout=5.0)
        thread.join(timeout=5.0)


# 全局模型实例
_model_instance: Optional[SAILVLModelCUDA] = None


def get_model() -> SAILVLModelCUDA:
    global _model_instance
    if _model_instance is None:
        _model_instance = SAILVLModelCUDA()
    return _model_instance


def init_model(
    model_path: str = "BytedanceDouyinContent/SAIL-VL2-8B-Thinking",
    quantization: str = "none",
    use_flash_attention: bool = True,
):
    """初始化全局模型实例

    Args:
        model_path: 模型路径
        quantization: 量化模式 - "none", "int8", "int4"
        use_flash_attention: 是否使用 Flash Attention 2
    """
    global _model_instance
    _model_instance = SAILVLModelCUDA(
        model_path,
        quantization=quantization,
        use_flash_attention=use_flash_attention,
    )
    _model_instance.load()
    return _model_instance
```

### 4.2 更新 api_server.py

修改 `api_server.py` 顶部的导入：

```python
# 替换这行
# from model_loader import get_model, init_model

# 为这行
from model_loader_cuda import get_model, init_model
```

修改启动代码：

```python
if __name__ == "__main__":
    # RTX 4090 推荐配置
    init_model(
        model_path="./models/SAIL-VL2-8B-Thinking",  # 或 HuggingFace ID
        quantization="int4",  # 推荐 INT4 量化
        use_flash_attention=True,  # 启用 Flash Attention
    )

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 五、启动脚本

### 5.1 创建 start_server_cuda.bat

```batch
@echo off
echo ========================================
echo SAIL-VL2-8B-Thinking Server (RTX 4090)
echo ========================================

:: 激活 conda 环境
call conda activate sailvl

:: 设置环境变量
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: 启动服务
python api_server.py

pause
```

### 5.2 创建 start_server_cuda.ps1 (PowerShell)

```powershell
Write-Host "========================================"
Write-Host "SAIL-VL2-8B-Thinking Server (RTX 4090)"
Write-Host "========================================"

# 激活 conda 环境
conda activate sailvl

# 设置环境变量
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 启动服务
python api_server.py
```

---

## 六、配置对比

### 6.1 显存占用估算

| 配置 | 显存占用 | 适用场景 |
|------|---------|---------|
| FP16 无量化 | ~16GB | 最高精度 |
| INT8 量化 | ~10GB | 平衡选择 |
| INT4 量化 | ~6GB | 最快速度 |

### 6.2 推荐配置

**最高性能 (推荐)**:
```python
init_model(
    quantization="int4",
    use_flash_attention=True,
)
```

**最高精度**:
```python
init_model(
    quantization="none",
    use_flash_attention=True,
)
```

**内存受限**:
```python
init_model(
    quantization="int4",
    use_flash_attention=True,
)
```

---

## 七、性能测试

### 7.1 创建 benchmark_cuda.py

```python
"""RTX 4090 性能测试"""
import time
import torch
import asyncio
from model_loader_cuda import SAILVLModelCUDA


async def benchmark():
    print("=" * 60)
    print("RTX 4090 性能测试")
    print("=" * 60)
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    configs = [
        ("FP16 无量化", "none"),
        ("INT8 量化", "int8"),
        ("INT4 量化", "int4"),
    ]

    results = []

    for name, quant in configs:
        print(f"\n测试: {name}")
        print("-" * 40)

        # 清理显存
        torch.cuda.empty_cache()

        # 加载模型
        model = SAILVLModelCUDA(
            quantization=quant,
            use_flash_attention=True,
        )

        start = time.time()
        model.load()
        load_time = time.time() - start

        # 显存使用
        vram = torch.cuda.memory_allocated() / 1024**3

        # 预热
        messages = [{"role": "user", "content": "Hello"}]
        await model.generate(messages, max_tokens=10, enable_thinking=False)

        # 性能测试
        test_messages = [{"role": "user", "content": "用100字介绍人工智能的发展历史。"}]

        start = time.time()
        response = await model.generate(
            test_messages,
            max_tokens=200,
            temperature=0.7,
            enable_thinking=False,
        )
        gen_time = time.time() - start

        # 计算 tokens
        output_tokens = len(model.tokenizer.encode(response))
        tokens_per_sec = output_tokens / gen_time

        print(f"  加载时间: {load_time:.1f}s")
        print(f"  显存占用: {vram:.2f} GB")
        print(f"  生成时间: {gen_time:.2f}s")
        print(f"  输出tokens: {output_tokens}")
        print(f"  速度: {tokens_per_sec:.1f} tokens/s")

        results.append({
            "name": name,
            "load_time": load_time,
            "vram": vram,
            "tokens_per_sec": tokens_per_sec,
        })

        # 清理
        del model
        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"{'配置':<20} {'显存(GB)':<10} {'速度(tok/s)':<15}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:<20} {r['vram']:<10.2f} {r['tokens_per_sec']:<15.1f}")


if __name__ == "__main__":
    asyncio.run(benchmark())
```

---

## 八、常见问题

### Q1: Flash Attention 安装失败

```powershell
# 尝试从源码编译
pip install ninja packaging
pip install flash-attn --no-build-isolation

# 或使用预编译版本
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

### Q2: CUDA Out of Memory

```python
# 使用更激进的量化
init_model(quantization="int4")

# 或减少 batch size / max_tokens
```

### Q3: 模型加载缓慢

```python
# 启用更快的加载
os.environ["SAFETENSORS_FAST_GPU"] = "1"
```

---

## 九、文件清单

迁移到 Windows 时需要的文件：

```
SAIL-VL2-8B-Thinking/
├── api_server.py           # API 服务器
├── model_loader_cuda.py    # CUDA 优化加载器 (新建)
├── config.py               # 配置文件
├── requirements_cuda.txt   # CUDA 依赖 (新建)
├── start_server_cuda.bat   # Windows 启动脚本 (新建)
├── benchmark_cuda.py       # 性能测试 (新建)
└── models/
    └── SAIL-VL2-8B-Thinking/  # 模型文件
```

### requirements_cuda.txt

```
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
accelerate>=0.27.0
bitsandbytes>=0.42.0
fastapi>=0.109.0
uvicorn>=0.27.0
httpx>=0.26.0
pillow>=10.0.0
sentencepiece
tiktoken
```

---

## 十、快速开始

```powershell
# 1. 克隆/复制项目
cd C:\Projects
# 复制项目文件...

# 2. 创建环境
conda create -n sailvl python=3.10 -y
conda activate sailvl

# 3. 安装依赖
pip install -r requirements_cuda.txt
pip install flash-attn --no-build-isolation

# 4. 下载模型
huggingface-cli download BytedanceDouyinContent/SAIL-VL2-8B-Thinking --local-dir ./models/SAIL-VL2-8B-Thinking

# 5. 启动服务
start_server_cuda.bat

# 6. 测试 API
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"sail-vl2-8b-thinking\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

预期 RTX 4090 性能：**80-120 tokens/s** (INT4 + Flash Attention)
