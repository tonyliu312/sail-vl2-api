"""
测试 quanto 量化加载和推理性能
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import asyncio
from model_loader import SAILVLModel

async def test_quantized_model():
    print("=" * 60)
    print("测试 quanto int8 量化")
    print("=" * 60)

    # 1. 加载量化模型
    print("\n1. 加载量化模型...")
    start_time = time.time()

    model = SAILVLModel(quantize=True)
    model.load()

    load_time = time.time() - start_time
    print(f"   模型加载耗时: {load_time:.2f}s")

    # 2. 测试推理
    print("\n2. 测试推理性能...")

    messages = [{"role": "user", "content": "你好，请用一句话介绍自己。"}]

    # 预热
    print("   预热中...")
    await model.generate(messages, max_tokens=10, enable_thinking=False)

    # 性能测试
    print("   正式测试...")
    start_time = time.time()
    response = await model.generate(
        messages,
        max_tokens=100,
        temperature=0.7,
        enable_thinking=False
    )
    gen_time = time.time() - start_time

    # 估算token数量（粗略）
    response_tokens = len(response) // 2  # 中文约2字符/token
    tokens_per_sec = response_tokens / gen_time if gen_time > 0 else 0

    print(f"\n3. 结果:")
    print(f"   响应: {response[:100]}...")
    print(f"   响应长度: {len(response)} 字符")
    print(f"   生成耗时: {gen_time:.2f}s")
    print(f"   估算速度: ~{tokens_per_sec:.1f} tokens/s")

    # 3. 流式测试
    print("\n4. 流式输出测试...")
    start_time = time.time()
    first_token_time = None
    chunk_count = 0

    async for chunk in model.generate_stream(
        messages,
        max_tokens=50,
        temperature=0.7,
        enable_thinking=False
    ):
        if first_token_time is None:
            first_token_time = time.time() - start_time
        chunk_count += 1
        print(chunk, end="", flush=True)

    stream_time = time.time() - start_time
    print(f"\n\n   首token延迟: {first_token_time:.2f}s")
    print(f"   流式输出 {chunk_count} 个 chunk，耗时 {stream_time:.2f}s")
    print(f"   流式速度: {chunk_count/stream_time:.1f} chunks/s")

    print("\n" + "=" * 60)
    print("量化测试完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_quantized_model())
