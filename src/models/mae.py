from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import TransformerBlock
from .vit import ViT, ViTConfig


@dataclass
class MAEConfig:
    d_model: int
    num_blocks: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attn_dropout: float
    patch_size: int
    image_size: int
    mask_ratio: float


class MaskedAutoEncoder(nn.Module):
    def __init__(self, config: MAEConfig):
        super().__init__()

        # ViT encoder
        self.vit_config = ViTConfig(
            d_model=config.d_model,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            patch_size=config.patch_size,
            image_size=config.image_size,
        )
        self.encoder = ViT(self.vit_config)

        self.num_patches = self.encoder.patch_embed.num_patches
        self.num_keep = int(self.num_patches * (1 - config.mask_ratio))

        # Decoder
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.d_model)
        )
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.decoder_pred = nn.Linear(
            config.d_model,
            config.patch_size * config.patch_size * 3,
        )

    def random_masking(self, x):
        batch_size, num_patches, dim = x.shape
        noise = torch.rand(batch_size, num_patches, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, : self.num_keep]

        x = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
        )

        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, : self.num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask, ids_restore

    def forward_encoder(self, x):
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed
        x, mask, ids_restore = self.random_masking(x)

        for block in self.encoder.encoder_blocks:
            x = block(x)

        x = self.encoder.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        batch_size, _, dim = x.shape
        num_mask = self.num_patches - self.num_keep

        x = torch.cat([x, self.mask_token.expand(batch_size, num_mask, -1)], dim=1)
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dim),
        )

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        x, mask, ids_restore = self.forward_encoder(x)
        x = self.forward_decoder(x, ids_restore)
        return x, mask
