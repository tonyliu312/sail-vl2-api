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
