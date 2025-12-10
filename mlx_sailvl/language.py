"""
MLX implementation of Qwen3 language model for SAIL-VL
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    head_dim: int = 128
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        llm_config = params.get("llm_config", params)
        return cls(
            hidden_size=llm_config.get("hidden_size", 4096),
            intermediate_size=llm_config.get("intermediate_size", 12288),
            num_hidden_layers=llm_config.get("num_hidden_layers", 36),
            num_attention_heads=llm_config.get("num_attention_heads", 32),
            num_key_value_heads=llm_config.get("num_key_value_heads", 8),
            vocab_size=llm_config.get("vocab_size", 152064),
            rms_norm_eps=llm_config.get("rms_norm_eps", 1e-5),
            rope_theta=llm_config.get("rope_theta", 1000000.0),
            max_position_embeddings=llm_config.get("max_position_embeddings", 40960),
            head_dim=llm_config.get("head_dim", 128),
            tie_word_embeddings=llm_config.get("tie_word_embeddings", False),
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RoPE(nn.Module):
    def __init__(self, dims: int, theta: float = 1000000.0):
        super().__init__()
        self.dims = dims
        self.theta = theta

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, 2)
        freqs = self._get_freqs(shape[-1], offset)
        cos_freqs = mx.cos(freqs)
        sin_freqs = mx.sin(freqs)
        x_r = x[..., 0] * cos_freqs - x[..., 1] * sin_freqs
        x_i = x[..., 0] * sin_freqs + x[..., 1] * cos_freqs
        return mx.stack([x_r, x_i], axis=-1).reshape(shape)

    def _get_freqs(self, dims: int, offset: int) -> mx.array:
        inv_freq = 1.0 / (self.theta ** (mx.arange(0, dims, 2) / dims))
        positions = mx.arange(offset, offset + 1)
        freqs = positions[:, None] * inv_freq[None, :]
        return freqs


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Qwen3 has separate q_norm and k_norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RoPE(self.head_dim, theta=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply Q/K normalization (Qwen3 specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)

        # Handle KV cache (store BEFORE head expansion)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)

        # Store cache before GQA expansion
        new_cache = (k, v)

        # Repeat KV heads if needed (GQA)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_expanded = mx.repeat(k, n_rep, axis=1)
            v_expanded = mx.repeat(v, n_rep, axis=1)
        else:
            k_expanded = k
            v_expanded = v

        # Scaled dot-product attention
        attn = (q @ k_expanded.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        output = (attn @ v_expanded).transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        return output, new_cache


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, cache


class Qwen3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[list]]:
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        # Create causal mask if not provided
        B, L, _ = h.shape
        if mask is None:
            # For generation with cache, we only need current position
            if cache is not None and cache[0] is not None:
                past_len = cache[0][0].shape[2]
                # No mask needed for single token generation with KV cache
                mask = None
            else:
                # Full causal mask for prefill
                mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
                mask = mask.astype(h.dtype)

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, mask, layer_cache)
            new_cache.append(c)

        return self.norm(h), new_cache


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[list]]:
        h, cache = self.model(input_ids, inputs_embeds, mask, cache)
        logits = self.lm_head(h)
        return logits, cache

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @staticmethod
    def sanitize(weights):
        """Convert weight names from PyTorch to MLX format"""
        new_weights = {}
        for k, v in weights.items():
            if "language_model" not in k and "model." not in k:
                continue

            # Remove prefix
            new_key = k.replace("language_model.", "")

            # Handle model layers
            new_key = new_key.replace("model.layers", "model.layers")
            new_key = new_key.replace("model.embed_tokens", "model.embed_tokens")
            new_key = new_key.replace("model.norm", "model.norm")

            new_weights[new_key] = v

        return new_weights
