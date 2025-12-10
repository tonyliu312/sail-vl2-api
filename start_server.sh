#!/bin/bash
# SAIL-VL2-8B-Thinking 服务启动脚本
# 自动修复配置并启动服务

cd "$(dirname "$0")"

# 不设置离线模式，由代码中的monkey-patch处理网络请求问题

# 修复配置文件中的 Qwen2 支持
for CONFIG_FILE in \
    "$HOME/.cache/huggingface/modules/transformers_modules/_228b6f5348c6791cf8321320868b3b70cee0165e/configuration_sailvl.py" \
    "$HOME/.cache/huggingface/modules/transformers_modules/BytedanceDouyinContent/SAIL_hyphen_VL2_hyphen_8B_hyphen_Thinking/228b6f5348c6791cf8321320868b3b70cee0165e/configuration_sailvl.py"
do
    if [ -f "$CONFIG_FILE" ]; then
        echo "正在修复配置文件: $CONFIG_FILE"
        sed -i '' "s/== 'Qwen3ForCausalLM'/in ('Qwen3ForCausalLM', 'Qwen2ForCausalLM')/g" "$CONFIG_FILE" 2>/dev/null || \
        sed -i "s/== 'Qwen3ForCausalLM'/in ('Qwen3ForCausalLM', 'Qwen2ForCausalLM')/g" "$CONFIG_FILE"
    fi
done

echo "启动API服务（离线模式）..."
python3 api_server.py
