#!/bin/bash

# SAIL-VL2-8B-Thinking API 服务启动脚本

# 默认配置
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

echo "========================================"
echo "  SAIL-VL2-8B-Thinking API Server"
echo "========================================"
echo ""
echo "配置信息:"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Workers: $WORKERS"
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "警告: PyTorch 未安装，请运行: pip install -r requirements.txt"
fi

if ! python3 -c "import transformers" &> /dev/null; then
    echo "警告: Transformers 未安装，请运行: pip install -r requirements.txt"
fi

# 检查 CUDA
echo ""
echo "GPU 信息:"
python3 -c "import torch; print(f'  - CUDA 可用: {torch.cuda.is_available()}'); print(f'  - GPU 数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else None" 2>/dev/null || echo "  - 无法检测 (PyTorch 未安装)"

echo ""
echo "启动服务..."
echo "API 文档: http://${HOST}:${PORT}/docs"
echo ""

# 启动服务
python3 -m uvicorn api_server:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
