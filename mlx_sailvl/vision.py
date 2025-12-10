"""
MLX implementation of SAILViT vision encoder (adapted from Apple AIMv2)
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 12
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    qkv_bias: bool = False
    use_bias: bool = False

    @classmethod
    def from_dict(cls, params):
        return cls(
            hidden_size=params.get("hidden_size", 1536),
            intermediate_size=params.get("intermediate_size", 4096),
            num_hidden_layers=params.get("num_hidden_layers", 24),
            num_attention_heads=params.get("num_attention_heads", 12),
            image_size=params.get("image_size", 448),
            patch_size=params.get("patch_size", 14),
            num_channels=params.get("num_channels", 3),
            rms_norm_eps=params.get("rms_norm_eps", 1e-5),
            attention_dropout=params.get("attention_dropout", 0.0),
            projection_dropout=params.get("projection_dropout", 0.0),
            qkv_bias=params.get("qkv_bias", False),
            use_bias=params.get("use_bias", False),
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: SiLU(fc1(x)) * fc3(x) -> fc2"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding with RMSNorm"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,  # Conv2d has bias in SAILViT
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) in MLX format
        x = self.proj(x)  # (B, H', W', hidden_size)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # (B, num_patches, hidden_size)
        x = self.norm(x)
        return x


class Preprocessor(nn.Module):
    """Preprocessor with patch embedding and positional embedding"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = PatchEmbed(config)
        self.pos_embed = mx.zeros((1, num_patches, config.hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        tokens = self.patchifier(x)
        B, N, C = tokens.shape
        tokens = tokens + self.pos_embed[:, :N]
        return tokens


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=config.use_bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """SAILViT Transformer block"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attn = Attention(config)
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config)
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class Trunk(nn.Module):
    """SAILViT Transformer trunk"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.blocks = [Block(config) for _ in range(config.num_hidden_layers)]
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        tokens: mx.array,
        mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[list]]:
        hidden_states = [] if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                hidden_states.append(tokens)
            tokens = block(tokens, mask)

        tokens = self.post_trunk_norm(tokens)

        if output_hidden_states:
            hidden_states.append(tokens)

        return tokens, hidden_states


class VisionModel(nn.Module):
    """SAILViT Vision Model"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.trunk = Trunk(config)

    def __call__(
        self,
        pixel_values: mx.array,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[list]]:
        # pixel_values: (B, H, W, C)
        x = self.preprocessor(pixel_values)
        x, hidden_states = self.trunk(x, output_hidden_states=output_hidden_states)
        return x, hidden_states
