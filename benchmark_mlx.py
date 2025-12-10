"""
Benchmark MLX SAIL-VL model inference performance
"""
import sys
import time
sys.path.insert(0, '/Users/tony/dev/SAIL-VL2-8B-Thinking')

import mlx.core as mx
from transformers import AutoTokenizer
from mlx_sailvl import Model, generate

MODEL_PATH = "/Users/tony/.cache/huggingface/hub/models--BytedanceDouyinContent--SAIL-VL2-8B-Thinking/snapshots/228b6f5348c6791cf8321320868b3b70cee0165e"

def main():
    print("=" * 60)
    print("MLX SAIL-VL Benchmark")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    start_time = time.time()
    model = Model.from_pretrained(MODEL_PATH)
    load_time = time.time() - start_time
    print(f"   Model loaded in {load_time:.2f}s")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"   Tokenizer loaded")

    # Test text generation
    print("\n3. Testing text generation...")
    prompt = "你好，请用一句话介绍自己。"

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"   Prompt: {prompt}")
    print(f"   Formatted: {text[:100]}...")

    # Tokenize
    inputs = tokenizer(text, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    print(f"   Input tokens: {input_ids.shape[1]}")

    # Warm-up run
    print("\n4. Warm-up run...")
    start_time = time.time()
    logits, _ = model(input_ids)
    mx.eval(logits)  # Force evaluation
    warmup_time = time.time() - start_time
    print(f"   Warm-up completed in {warmup_time:.2f}s")

    # Benchmark prefill (first token)
    print("\n5. Benchmarking prefill (first token latency)...")
    num_runs = 3
    prefill_times = []

    for i in range(num_runs):
        start_time = time.time()
        logits, cache = model(input_ids)
        mx.eval(logits)
        prefill_times.append(time.time() - start_time)
        print(f"   Run {i+1}: {prefill_times[-1]:.3f}s")

    avg_prefill = sum(prefill_times) / len(prefill_times)
    print(f"   Average prefill: {avg_prefill:.3f}s")

    # Benchmark decode (token generation)
    print("\n6. Benchmarking decode (token/s)...")

    # Generate tokens
    num_tokens = 50
    generated_tokens = []
    decode_times = []

    # Initial prefill
    logits, cache = model(input_ids)
    mx.eval(logits)

    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    generated_tokens.append(next_token.item())

    print(f"   Generating {num_tokens} tokens...")
    for i in range(num_tokens - 1):
        start_time = time.time()
        logits, cache = model(
            mx.array([[generated_tokens[-1]]]),
            cache=cache
        )
        mx.eval(logits)
        decode_time = time.time() - start_time
        decode_times.append(decode_time)

        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        generated_tokens.append(next_token.item())

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    tokens_per_sec = 1.0 / avg_decode if avg_decode > 0 else 0

    print(f"   Generated {len(generated_tokens)} tokens")
    print(f"   Average decode time: {avg_decode*1000:.2f}ms/token")
    print(f"   Throughput: {tokens_per_sec:.2f} tokens/s")

    # Decode and print result
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\n7. Generated text:")
    print(f"   {generated_text}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary (MLX)")
    print("=" * 60)
    print(f"   Model load time:    {load_time:.2f}s")
    print(f"   Prefill latency:    {avg_prefill:.3f}s ({input_ids.shape[1]} tokens)")
    print(f"   Decode throughput:  {tokens_per_sec:.2f} tokens/s")
    print(f"   First token time:   {avg_prefill:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
