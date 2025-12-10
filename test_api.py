#!/usr/bin/env python3
"""
API 测试脚本
使用 OpenAI Python SDK 测试 API 服务
"""
import base64
import httpx


def test_with_httpx():
    """使用 httpx 测试 API"""
    base_url = "http://localhost:8000"

    # 测试健康检查
    print("=" * 50)
    print("测试健康检查")
    print("=" * 50)
    response = httpx.get(f"{base_url}/health")
    print(f"状态: {response.json()}")
    print()

    # 测试模型列表
    print("=" * 50)
    print("测试模型列表")
    print("=" * 50)
    response = httpx.get(f"{base_url}/v1/models")
    print(f"模型列表: {response.json()}")
    print()

    # 测试纯文本对话
    print("=" * 50)
    print("测试纯文本对话")
    print("=" * 50)
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "sail-vl2-8b-thinking",
            "messages": [
                {"role": "user", "content": "中国的首都是哪里？"}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "enable_thinking": True
        },
        timeout=300.0
    )
    result = response.json()
    print(f"响应: {result['choices'][0]['message']['content']}")
    print()

    # 测试带图片的对话（需要提供实际图片URL）
    print("=" * 50)
    print("测试视觉问答（示例）")
    print("=" * 50)
    print("提示: 要测试图片功能，请修改此脚本中的图片URL")
    print()

    # 示例：带图片URL的请求格式
    example_request = {
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
        "max_tokens": 1024
    }
    print(f"图片请求格式示例:")
    print(f"{example_request}")


def test_with_openai_sdk():
    """使用 OpenAI SDK 测试 API"""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI SDK 未安装，跳过此测试")
        print("安装命令: pip install openai")
        return

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  # 本地服务不需要 API key
    )

    print("=" * 50)
    print("使用 OpenAI SDK 测试")
    print("=" * 50)

    # 纯文本测试
    response = client.chat.completions.create(
        model="sail-vl2-8b-thinking",
        messages=[
            {"role": "user", "content": "1+1等于多少？请详细解释。"}
        ],
        max_tokens=512
    )
    print(f"响应: {response.choices[0].message.content}")
    print()

    # 流式测试
    print("流式响应测试:")
    stream = client.chat.completions.create(
        model="sail-vl2-8b-thinking",
        messages=[
            {"role": "user", "content": "写一首短诗"}
        ],
        max_tokens=256,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("SAIL-VL2-8B-Thinking API 测试")
    print("=" * 50)
    print("确保服务已启动: python api_server.py")
    print("=" * 50)
    print()

    test_with_httpx()
    print()
    test_with_openai_sdk()
