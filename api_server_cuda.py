"""
SAIL-VL2-8B-Thinking OpenAI 兼容 API 服务器 (CUDA/RTX 4090 版本)
"""
import os
import time
import uuid
import json
from typing import Optional, List, Union, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 使用 CUDA 优化的模型加载器
from model_loader_cuda import get_model, init_model

app = FastAPI(
    title="SAIL-VL2-8B-Thinking API (CUDA)",
    description="OpenAI 兼容的视觉语言模型 API - RTX 4090 优化版",
    version="1.0.0"
)


# 请求/响应模型
class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]


class ChatCompletionRequest(BaseModel):
    model: str = "sail-vl2-8b-thinking"
    messages: List[Message]
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False
    enable_thinking: bool = True


class Choice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class DeltaChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[DeltaChoice]


@app.get("/health")
async def health_check():
    """健康检查"""
    import torch
    return {
        "status": "healthy",
        "model": "sail-vl2-8b-thinking",
        "backend": "cuda",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "sail-vl2-8b-thinking",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bytedance",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天完成接口"""
    model = get_model()

    # 转换消息格式
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            content_list = []
            for item in msg.content:
                if item.type == "text":
                    content_list.append({"type": "text", "text": item.text})
                elif item.type == "image_url" and item.image_url:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": item.image_url.url}
                    })
            messages.append({"role": msg.role, "content": content_list})

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            stream_response(
                model, messages, request, request_id, created
            ),
            media_type="text/event-stream"
        )
    else:
        # 非流式响应
        response = await model.generate(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            enable_thinking=request.enable_thinking,
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message={"role": "assistant", "content": response},
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,  # 简化处理
                completion_tokens=len(response) // 4,
                total_tokens=len(response) // 4
            )
        )


async def stream_response(
    model,
    messages: list,
    request: ChatCompletionRequest,
    request_id: str,
    created: int
) -> AsyncGenerator[str, None]:
    """流式响应生成器"""
    try:
        async for chunk in model.generate_stream(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            enable_thinking=request.enable_thinking,
        ):
            response = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    DeltaChoice(
                        index=0,
                        delta={"content": chunk},
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {response.model_dump_json()}\n\n"

        # 发送结束标记
        final_response = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                DeltaChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_response = {"error": str(e)}
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    # RTX 4090 推荐配置
    # quantization: "none" (最高精度), "int8" (平衡), "int4" (最快)
    # use_flash_attention: True (推荐)

    print("=" * 60)
    print("SAIL-VL2-8B-Thinking API Server (CUDA)")
    print("=" * 60)

    # 从环境变量读取配置
    model_path = os.getenv("MODEL_PATH", "BytedanceDouyinContent/SAIL-VL2-8B-Thinking")
    quantization = os.getenv("QUANTIZATION", "int4")  # 默认 INT4 量化
    use_flash_attention = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"模型路径: {model_path}")
    print(f"量化模式: {quantization}")
    print(f"Flash Attention: {use_flash_attention}")
    print(f"服务地址: http://{host}:{port}")
    print("=" * 60)

    # 初始化模型
    init_model(
        model_path=model_path,
        quantization=quantization,
        use_flash_attention=use_flash_attention,
    )

    # 启动服务
    uvicorn.run(app, host=host, port=port)
