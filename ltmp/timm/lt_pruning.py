import math
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, LayerScale, VisionTransformer

from .threshold_masking import ThresholdMasker, softmax_with_mask
from .utils import create_vision_transformer


class LTPBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1.eval()
        self.attn = LTPAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm2.eval()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Add learned threshold masking modules for pruning
        self.prune_masker = ThresholdMasker()
        nn.init.constant_(self.prune_masker.threshold, 0.0)

    def forward(self, x, mask):
        x_attn, importance_scores = self.attn(self.norm1(x), mask)
        new_mask = self.prune_masker(importance_scores)
        mask[mask.to(torch.bool)] = new_mask[mask.to(torch.bool)]
        x = x + self.drop_path1(self.ls1((x_attn)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, mask


class LTPAttention(Attention):
    def forward(self, x, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = softmax_with_mask(attn, mask, self.training)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Calculate importance scores
        importance_scores = torch.mean(attn, dim=(1, 2))
        importance_scores[..., 0] = math.inf

        # Return k as well here
        return x, importance_scores


class LTPVisionTransformer(VisionTransformer):
    def __init__(self, tau=0.1, **kwargs):
        super().__init__(**kwargs)
        self.masks = []
        for block in self.blocks:
            block.prune_masker.tau = tau

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        self.masks = []
        m = torch.ones(x.shape[:-1], device=x.device)
        for block in self.blocks:
            x, m = block(x, m)
            self.masks.append(m.clone())
        x = self.norm(x)
        return x


## Register all vision transformer variants to which you want to apply ltp here.
# VIT
@register_model
def ltp_vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=LTPBlock, **kwargs)
    model = create_vision_transformer(
        LTPVisionTransformer, "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def ltp_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=LTPBlock, **kwargs)
    model = create_vision_transformer(
        LTPVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


# DEIT
@register_model
def ltp_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=LTPBlock, **kwargs)
    model = create_vision_transformer(
        LTPVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def ltp_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=LTPBlock, **kwargs)
    model = create_vision_transformer(
        LTPVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
