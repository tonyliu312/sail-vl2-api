"""
SAIL-VL2-8B-Thinking 模型加载器
"""
import os
# 设置离线模式环境变量（必须在导入transformers之前设置）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, TextIteratorStreamer
from PIL import Image
import io
import base64
from typing import Optional, List, Union, AsyncGenerator
import httpx
import asyncio
from threading import Thread

# Monkey-patch 绕过 transformers 的网络检查
try:
    import transformers.tokenization_utils_base as tub
    # 让 is_base_mistral 总是返回 False，避免网络请求
    tub.is_base_mistral = lambda x: False
except Exception:
    pass


class SAILVLModel:
    def __init__(self, model_path: str = "BytedanceDouyinContent/SAIL-VL2-8B-Thinking"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None

    def _get_device_and_dtype(self):
        """检测最佳设备和数据类型"""
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16
        elif torch.backends.mps.is_available():
            # Apple Silicon MPS 加速
            return torch.device("mps"), torch.float16  # MPS 不支持 bfloat16
        else:
            return torch.device("cpu"), torch.float32

    def load(self):
        """加载模型、tokenizer和processor"""
        print(f"正在加载模型: {self.model_path}")

        # 使用原始model_path (HuggingFace model ID)，这会加载已修复的模块
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.device, self.dtype = self._get_device_and_dtype()
        print(f"使用设备: {self.device}, 数据类型: {self.dtype}")

        # SAILVLModel 目前不支持 SDPA，只能使用 eager
        # 未来如果模型支持 SDPA，可以启用以获得 30-50% 的性能提升
        attn_impl = "eager"
        print(f"Attention 实现: {attn_impl}")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
            low_cpu_mem_usage=True,  # 减少 CPU 内存使用
        )

        # 移动模型到设备
        self.model = self.model.to(self.device)

        self.model.eval()

        # 尝试使用 torch.compile 优化（实验性）
        if self.device.type == "mps":
            try:
                # MPS 上使用 inductor 或 eager 后端
                # 注意：torch.compile 在 MPS 上可能不稳定
                # self.model = torch.compile(self.model, mode="reduce-overhead")
                # print("✓ torch.compile 优化已启用")
                pass
            except Exception as e:
                print(f"torch.compile 不可用: {e}")

        print(f"模型加载完成，设备: {self.device}")

    async def load_image_from_url(self, url: str) -> Image.Image:
        """从URL加载图片"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

    def load_image_from_base64(self, base64_str: str) -> Image.Image:
        """从base64加载图片"""
        # 处理 data URI 格式
        if base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    async def process_content(self, content: Union[str, List]) -> tuple[str, Optional[Image.Image]]:
        """
        处理OpenAI格式的content，返回文本和图片
        content可以是字符串或者包含text/image_url的列表
        """
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
        """
        生成响应

        Args:
            messages: OpenAI格式的消息列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p采样参数
            enable_thinking: 是否启用思考模式
        """
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

            # 在最后一条用户消息添加思考提示
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

        # 处理输入 - 使用实例的设备和数据类型
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 移动到正确的设备并转换数据类型
        inputs = inputs.to(self.device)
        if self.dtype != torch.float32:
            # 仅对浮点张量转换数据类型
            for key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    inputs[key] = val.to(self.dtype)

        # 生成配置 - 添加推理优化参数
        # 注意：use_cache 由模型内部处理，不需要外部传递
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # 生成响应 - 使用 inference_mode 比 no_grad 更快
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # 解码响应
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 清理响应
        if '<|im_end|>' in response:
            response = response.split('<|im_end|>')[0].strip()

        # 提取助手回复部分
        if '<|im_start|>assistant' in response:
            response = response.split('<|im_start|>assistant')[-1].strip()

        # 处理思考模式的输出
        thinking_content = None
        final_content = response

        if '<think>' in response:
            import re
            # 提取思考内容
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                # 移除思考部分，保留最终答案
                final_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            else:
                # 思考标签未闭合（被截断），提取思考后的内容
                parts = response.split('<think>')
                if len(parts) > 1:
                    thinking_content = parts[1].strip()
                    final_content = ""  # 思考被截断，没有最终答案

        # 如果启用思考模式且有思考内容，返回带思考的格式
        if enable_thinking and thinking_content:
            if final_content:
                return f"<think>\n{thinking_content}\n</think>\n\n{final_content}"
            else:
                return f"<think>\n{thinking_content}\n</think>"

        # 如果没有最终内容但有思考内容，返回思考内容作为响应
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
        """流式生成响应 - 真正的异步流式输出"""
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

            # 在最后一条用户消息添加思考提示
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

        # 处理输入 - 使用实例的设备和数据类型
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 移动到正确的设备并转换数据类型
        inputs = inputs.to(self.device)
        if self.dtype != torch.float32:
            for key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor) and val.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    inputs[key] = val.to(self.dtype)

        # 创建流式生成器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0  # 增加超时时间
        )

        # 生成配置 - 添加推理优化参数
        # 注意：use_cache 由模型内部处理，不需要外部传递
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # 在后台线程中运行生成 - 使用 inference_mode 比 no_grad 更快
        def generate_in_thread():
            with torch.inference_mode():
                self.model.generate(**inputs, **gen_kwargs)

        thread = Thread(target=generate_in_thread)
        thread.start()

        # 使用队列在线程间安全地传递数据
        import queue
        token_queue = queue.Queue()
        SENTINEL = object()  # 用于标记迭代结束

        def stream_to_queue():
            """从 streamer 读取数据并放入队列"""
            try:
                for text in streamer:
                    token_queue.put(text)
            except Exception as e:
                token_queue.put(e)
            finally:
                token_queue.put(SENTINEL)

        # 启动队列填充线程
        queue_thread = Thread(target=stream_to_queue)
        queue_thread.start()

        # 从队列中异步获取数据
        loop = asyncio.get_event_loop()

        while True:
            try:
                # 在线程池中运行阻塞的队列获取操作
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
                # 队列超时，检查线程是否还在运行
                if not thread.is_alive() and not queue_thread.is_alive():
                    break
                continue
            except Exception as e:
                # 其他错误
                break

        # 等待线程结束
        queue_thread.join(timeout=5.0)
        thread.join(timeout=5.0)


# 全局模型实例
_model_instance: Optional[SAILVLModel] = None


def get_model() -> SAILVLModel:
    """获取全局模型实例"""
    global _model_instance
    if _model_instance is None:
        _model_instance = SAILVLModel()
    return _model_instance


def init_model(model_path: str = "BytedanceDouyinContent/SAIL-VL2-8B-Thinking"):
    """初始化全局模型实例"""
    global _model_instance
    _model_instance = SAILVLModel(model_path)
    _model_instance.load()
    return _model_instance
