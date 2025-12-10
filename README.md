# SAIL-VL2-8B-Thinking API Server

OpenAI 兼容的视觉语言模型 API 服务，基于字节跳动的 [SAIL-VL2-8B-Thinking](https://huggingface.co/BytedanceDouyinContent/SAIL-VL2-8B-Thinking) 模型。

## 功能特点

- OpenAI API 兼容格式
- 支持图像理解（URL 或 Base64）
- 支持流式输出
- 支持思考模式（Chain of Thought）
- 支持 CUDA、MPS (Apple Silicon) 和 CPU

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

模型会在首次启动时自动从 HuggingFace 下载，或者手动下载：

```bash
huggingface-cli download BytedanceDouyinContent/SAIL-VL2-8B-Thinking
```

### 3. 启动服务

```bash
python api_server.py
# 或使用启动脚本
./start_server.sh
```

服务默认运行在 `http://localhost:8000`

## API 使用

### 聊天完成

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sail-vl2-8b-thinking",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下自己"
      }
    ]
  }'
```

### 图像理解

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sail-vl2-8b-thinking",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
          },
          {
            "type": "text",
            "text": "描述这张图片"
          }
        ]
      }
    ],
    "enable_thinking": true
  }'
```

### 流式输出

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sail-vl2-8b-thinking",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天完成（支持流式） |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger API 文档 |

## 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model` | string | sail-vl2-8b-thinking | 模型名称 |
| `messages` | array | - | 消息列表 |
| `max_tokens` | int | 2048 | 最大生成 token 数 |
| `temperature` | float | 0.7 | 采样温度 |
| `top_p` | float | 0.9 | Top-p 采样 |
| `stream` | bool | false | 是否流式输出 |
| `enable_thinking` | bool | true | 是否启用思考模式 |

## 项目结构

```
.
├── api_server.py      # FastAPI 服务器
├── model_loader.py    # 模型加载和推理
├── config.py          # 配置管理
├── test_api.py        # API 测试脚本
├── start_server.sh    # 启动脚本
├── run.sh             # 运行脚本
└── requirements.txt   # Python 依赖
```

## 硬件要求

- **GPU 推荐**: NVIDIA GPU (16GB+ VRAM) 或 Apple Silicon (M1/M2/M3/M4/M5)
- **内存**: 32GB+ RAM
- **存储**: 约 16GB（模型权重）

## 性能说明

- Apple Silicon (MPS): 首 token ~0.7s，生成速度 ~12 chars/s
- NVIDIA GPU: 性能更佳，具体取决于显卡型号

## License

本项目代码采用 MIT License。模型权重请遵循 [SAIL-VL2 模型许可证](https://huggingface.co/BytedanceDouyinContent/SAIL-VL2-8B-Thinking)。

## 致谢

- [ByteDance](https://github.com/bytedance) - SAIL-VL2 模型
- [HuggingFace](https://huggingface.co/) - Transformers 框架
