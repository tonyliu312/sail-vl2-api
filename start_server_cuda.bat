@echo off
chcp 65001 > nul
echo ========================================
echo SAIL-VL2-8B-Thinking Server (RTX 4090)
echo ========================================

:: 激活 conda 环境 (如果使用)
:: call conda activate sailvl

:: 设置环境变量
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: 启动服务
python api_server_cuda.py

pause
