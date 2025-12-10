"""
OpenAI 兼容的 API 服务器
支持 /v1/chat/completions 端点
"""
# 必须在最开始设置环境变量和monkey-patch，在任何其他导入之前
import os
os.environ["HF_HUB_OFFLINE"] = "0"  # 不使用离线模式，而是通过monkey-patch绕过

# Monkey-patch 绕过 transformers 的网络检查 - 必须在导入transformers之前
import transformers.tokenization_utils_base as tub
original_is_base_mistral = getattr(tub, 'is_base_mistral', None)
tub.is_base_mistral = lambda x: False

import asyncio
import time
import uuid
from typing import Optional, List, Union, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from model_loader import init_model, get_model


# ============ Pydantic 模型定义 ============

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str = "sail-vl2-8b-thinking"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1)
    stream: Optional[bool] = False
    # 自定义参数
    enable_thinking: Optional[bool] = True


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
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
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============ FastAPI 应用 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    print("正在初始化模型...")
    init_model()
    print("模型初始化完成!")
    yield
    # 关闭时清理资源
    print("正在关闭服务...")


app = FastAPI(
    title="SAIL-VL2-8B-Thinking API",
    description="OpenAI 兼容的视觉语言模型 API 服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ API 端点 ============

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出可用模型"""
    return ModelList(
        data=[
            ModelInfo(
                id="sail-vl2-8b-thinking",
                created=int(time.time()),
                owned_by="bytedance"
            )
        ]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """获取模型信息"""
    if model_id != "sail-vl2-8b-thinking":
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return ModelInfo(
        id="sail-vl2-8b-thinking",
        created=int(time.time()),
        owned_by="bytedance"
    )


async def generate_stream(
    request: ChatCompletionRequest,
    response_id: str,
    created: int
) -> AsyncGenerator[str, None]:
    """生成流式响应 - 真正的流式输出"""
    model = get_model()

    # 转换消息格式
    messages = [msg.model_dump() for msg in request.messages]

    try:
        # 使用真正的流式生成
        async for chunk_text in model.generate_stream(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            enable_thinking=request.enable_thinking,
        ):
            if chunk_text:  # 只发送非空内容
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta={"content": chunk_text},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # 发送结束标记
        final_chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """创建聊天完成"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # 流式响应
    if request.stream:
        return StreamingResponse(
            generate_stream(request, response_id, created),
            media_type="text/event-stream"
        )

    # 非流式响应
    model = get_model()

    # 转换消息格式
    messages = [msg.model_dump() for msg in request.messages]

    try:
        response_text = await model.generate(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            enable_thinking=request.enable_thinking,
        )

        # 估算token数（简单估算，实际应该使用tokenizer）
        prompt_tokens = sum(
            len(str(msg.content)) // 4 for msg in request.messages
        )
        completion_tokens = len(response_text) // 4

        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": get_model().model is not None}


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "SAIL-VL2-8B-Thinking API Server",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
