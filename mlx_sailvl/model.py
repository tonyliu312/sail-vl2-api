"""
MLX implementation of SAIL-VL model
"""
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .vision import VisionConfig, VisionModel
from .language import TextConfig, LanguageModel


@dataclass
class ModelConfig:
    vision_config: VisionConfig
    text_config: TextConfig
    model_type: str = "sailvl"
    downsample_ratio: float = 0.5
    select_layer: int = -4
    img_context_token_id: int = 151671
    num_image_token: int = 256  # (448/14)^2 * 0.5^2

    @classmethod
    def from_dict(cls, params):
        vision_config = VisionConfig.from_dict(params.get("vision_config", {}))
        text_config = TextConfig.from_dict(params)

        # Calculate num_image_token
        image_size = params.get("force_image_size", 448)
        patch_size = params.get("vision_config", {}).get("patch_size", 14)
        downsample_ratio = params.get("downsample_ratio", 0.5)
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        return cls(
            vision_config=vision_config,
            text_config=text_config,
            model_type=params.get("model_type", "sailvl"),
            downsample_ratio=downsample_ratio,
            select_layer=params.get("select_layer", -4),
            num_image_token=num_image_token,
        )


class MLPProjector(nn.Module):
    """MLP projector to map vision features to language model hidden space"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

        # Calculate projector input dimension after pixel shuffle
        vit_hidden = config.vision_config.hidden_size
        downsample_factor = int(1 / config.downsample_ratio)
        projector_input_dim = vit_hidden * (downsample_factor ** 2)

        self.mlp1 = MLPProjector(projector_input_dim, config.text_config.hidden_size)
        self.downsample_ratio = config.downsample_ratio

    def pixel_shuffle(self, x: mx.array, scale_factor: float = 0.5) -> mx.array:
        """Pixel shuffle to downsample spatial resolution while increasing channels"""
        n, h, w, c = x.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        new_c = int(c / (scale_factor * scale_factor))

        # Reshape: N, H, W, C -> N, H, W*scale, C/scale
        x = x.reshape(n, h, int(w * scale_factor), int(c / scale_factor))
        # Transpose: N, H, W*scale, C/scale -> N, W*scale, H, C/scale
        x = x.transpose(0, 2, 1, 3)
        # Reshape: N, W*scale, H, C/scale -> N, W*scale, H*scale, C/(scale^2)
        x = x.reshape(n, new_w, new_h, new_c)
        # Transpose back: N, W*scale, H*scale, C/(scale^2) -> N, H*scale, W*scale, C/(scale^2)
        x = x.transpose(0, 2, 1, 3)

        return x

    def extract_feature(self, pixel_values: mx.array) -> mx.array:
        """Extract and project vision features"""
        # Get vision features
        vit_embeds, hidden_states = self.vision_model(
            pixel_values,
            output_hidden_states=(self.config.select_layer != -1)
        )

        if self.config.select_layer != -1 and hidden_states is not None:
            vit_embeds = hidden_states[self.config.select_layer]

        # Reshape to spatial format
        B, N, C = vit_embeds.shape
        h = w = int(N ** 0.5)
        vit_embeds = vit_embeds.reshape(B, h, w, C)

        # Apply pixel shuffle
        vit_embeds = self.pixel_shuffle(vit_embeds, self.downsample_ratio)

        # Flatten back to sequence
        vit_embeds = vit_embeds.reshape(B, -1, vit_embeds.shape[-1])

        # Project to language model dimension
        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ) -> mx.array:
        """Get input embeddings with vision features merged"""
        if pixel_values is None:
            return self.language_model.get_input_embeddings()(input_ids)

        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Get vision features
        vit_embeds = self.extract_feature(pixel_values)

        # Merge vision features into text embeddings at image token positions
        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)

        # Find image token positions
        mask = (input_ids_flat == self.config.img_context_token_id)

        if mx.sum(mask) > 0:
            # Replace image tokens with vision features
            vit_embeds_flat = vit_embeds.reshape(-1, C)
            num_vision_tokens = min(mx.sum(mask), vit_embeds_flat.shape[0])

            # Use advanced indexing to replace
            indices = mx.where(mask)[0][:num_vision_tokens]
            for i, idx in enumerate(indices.tolist()):
                if i < vit_embeds_flat.shape[0]:
                    inputs_embeds = mx.concatenate([
                        inputs_embeds[:idx],
                        vit_embeds_flat[i:i+1],
                        inputs_embeds[idx+1:]
                    ], axis=0)

        inputs_embeds = inputs_embeds.reshape(B, N, C)
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[list]]:
        """Forward pass"""
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """Load model from pretrained weights"""
        path = Path(path_or_hf_repo)
        # Check if it's a local path (exists or starts with / or ~)
        is_local = path.exists() or path_or_hf_repo.startswith('/') or path_or_hf_repo.startswith('~')
        if not is_local:
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        config = ModelConfig.from_dict(model_config)
        model = Model(config)

        # Load weights
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        # Sanitize weight names
        weights = model.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        """Convert PyTorch weight names to MLX format"""
        new_weights = {}

        for k, v in weights.items():
            new_key = k

            # Vision model weights - SAILViT structure
            # vision_model.preprocessor.patchifier.proj.weight/bias
            # vision_model.preprocessor.patchifier.norm.weight
            # vision_model.preprocessor.pos_embed
            # vision_model.trunk.blocks.X.attn.qkv/proj.weight
            # vision_model.trunk.blocks.X.norm_1/norm_2.weight
            # vision_model.trunk.blocks.X.mlp.fc1/fc2/fc3.weight
            # vision_model.trunk.post_trunk_norm.weight
            if "vision_model" in k:
                # Conv weight transposition: PyTorch (out, in, H, W) -> MLX (out, H, W, in)
                if "patchifier.proj.weight" in k and v.ndim == 4:
                    v = v.transpose(0, 2, 3, 1)
                new_key = k  # Names already match our MLX structure

            # Language model weights
            elif "language_model" in k:
                new_key = k

            # MLP projector weights
            # mlp1.0 -> mlp1.norm (LayerNorm)
            # mlp1.1 -> mlp1.fc1 (Linear)
            # mlp1.3 -> mlp1.fc2 (Linear, index 3 because GELU is index 2)
            elif "mlp1" in k:
                new_key = k
                new_key = new_key.replace("mlp1.0", "mlp1.norm")
                new_key = new_key.replace("mlp1.1", "mlp1.fc1")
                new_key = new_key.replace("mlp1.3", "mlp1.fc2")

            new_weights[new_key] = v

        return new_weights


def generate(
    model: Model,
    tokenizer,
    prompt: str,
    pixel_values: Optional[mx.array] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate text from the model"""
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])

    # Generate
    cache = None
    generated = []

    for _ in range(max_tokens):
        logits, cache = model(
            input_ids if len(generated) == 0 else mx.array([[generated[-1]]]),
            pixel_values if len(generated) == 0 else None,
            cache=cache
        )

        # Sample next token
        if temperature > 0:
            logits = logits[:, -1, :] / temperature
            # Top-p sampling
            probs = mx.softmax(logits, axis=-1)
            sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
            cumsum = mx.cumsum(sorted_probs, axis=-1)
            mask = cumsum > top_p
            # Keep at least one token
            mask = mx.concatenate([mx.zeros((1, 1)), mask[:, :-1]], axis=1)
            probs = mx.where(mask, mx.zeros_like(probs), probs)
            probs = probs / mx.sum(probs, axis=-1, keepdims=True)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        else:
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

        next_token = next_token.item()
        generated.append(next_token)

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)
