"""
配置文件
"""
import os

# 模型配置
MODEL_PATH = os.getenv("MODEL_PATH", "BytedanceDouyinContent/SAIL-VL2-8B-Thinking")

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# 生成配置默认值
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))

# 是否默认启用思考模式
DEFAULT_ENABLE_THINKING = os.getenv("DEFAULT_ENABLE_THINKING", "true").lower() == "true"
