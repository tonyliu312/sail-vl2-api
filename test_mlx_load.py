"""
Test MLX SAIL-VL model loading
"""
import sys
sys.path.insert(0, '/Users/tony/dev/SAIL-VL2-8B-Thinking')

import mlx.core as mx
from mlx_sailvl import Model, ModelConfig

# Model path
MODEL_PATH = "/Users/tony/.cache/huggingface/hub/models--BytedanceDouyinContent--SAIL-VL2-8B-Thinking/snapshots/228b6f5348c6791cf8321320868b3b70cee0165e"

print("Loading MLX SAIL-VL model...")
print(f"Model path: {MODEL_PATH}")

try:
    model = Model.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")
    print(f"Model config: {model.config}")

    # Test with dummy input
    print("\nTesting with dummy input...")
    dummy_input = mx.array([[1, 2, 3, 4, 5]])
    logits, cache = model(dummy_input)
    print(f"Output shape: {logits.shape}")
    print("Inference test passed!")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
