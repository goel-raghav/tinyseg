from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import TransformerBlock


@dataclass
class ViTConfig:
    d_model: int
    num_blocks: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attn_dropout: float
    patch_size: int
    image_size: int


class PatchEmbed(nn.Module):
    def __init__(self, d_model, patch_size, image_size):
        assert image_size % patch_size == 0, (
            "image_size must be divisible by patch_size"
        )

        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            config.d_model, config.patch_size, config.image_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, config.d_model)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    config.attn_dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        return x
