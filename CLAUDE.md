# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是 SAIL-VL2-8B-Thinking 视觉语言模型的 OpenAI 兼容 API 部署服务。该模型由字节跳动开发，支持图像理解和推理思考功能。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python api_server.py
# 或使用脚本
./run.sh

# 测试 API
python test_api.py

# 使用环境变量配置
HOST=0.0.0.0 PORT=8000 ./run.sh
```

## 架构说明

- `model_loader.py`: 模型加载和推理核心逻辑，包含 `SAILVLModel` 类
- `api_server.py`: FastAPI 实现的 OpenAI 兼容 API 服务器
- `config.py`: 环境变量和配置管理
- `test_api.py`: API 测试脚本

## API 端点

- `POST /v1/chat/completions` - 聊天完成（支持流式和非流式）
- `GET /v1/models` - 列出可用模型
- `GET /health` - 健康检查

## 请求格式

支持 OpenAI 标准格式，包括多模态输入（图片 URL 或 base64）：

```python
{
    "model": "sail-vl2-8b-thinking",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": "描述这张图片"}
            ]
        }
    ],
    "enable_thinking": true  # 启用思考模式（可选）
}
```

## 注意事项

- 模型需要 GPU（推荐）或大内存 CPU 运行
- 使用 bfloat16 精度以节省显存
- `enable_thinking` 参数控制是否启用 CoT 思考模式
