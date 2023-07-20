import math
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, LayerScale, VisionTransformer

from .threshold_masking import InferenceThresholdMasker
from .utils import create_vision_transformer


class InferenceLTMBlock(nn.Module):
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
        self.attn = InferenceLTMAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Add learned threshold masking modules for merging
        self.merge_masker = InferenceThresholdMasker()
        nn.init.constant_(self.merge_masker.threshold, 1)

    def forward(self, x, size=None) -> torch.Tensor:
        if size is None:
            size = torch.ones_like(x[..., 0, None])
        x_attn, metric = self.attn(self.norm1(x), size)
        x = x + self.drop_path1(self.ls1((x_attn)))

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        scores[..., 0, :] = -math.inf

        b, _, c = x.shape
        node_max, node_idx = scores.max(dim=-1)
        merge_mask = self.merge_masker(node_max).bool()
        unm_mask = ~merge_mask

        x = x * size
        src_x, dst_x = x[..., ::2, :], x[..., 1::2, :]
        src_s, dst_s = size[..., ::2, :], size[..., 1::2, :]

        unm_x = src_x[unm_mask].unsqueeze(0)
        unm_s = src_s[unm_mask].unsqueeze(0)
        _, merge_indices = merge_mask.nonzero(as_tuple=True)
        merge_indices = merge_indices.unsqueeze(0)
        dst_idx = node_idx.gather(dim=-1, index=merge_indices)
        if dst_idx.numel() > 0:
            dst_x = dst_x.scatter_reduce(
                1, dst_idx.unsqueeze(-1).expand(dst_idx.shape + (c,)), src_x[merge_mask].unsqueeze(0), reduce="sum"
            )
            dst_s = dst_s.scatter_reduce(
                1, dst_idx.unsqueeze(-1).expand(dst_idx.shape + (1,)), src_s[merge_mask].unsqueeze(0), reduce="sum"
            )
        x = torch.cat([unm_x, dst_x], dim=1)
        size = torch.cat([unm_s, dst_s], dim=1)
        x = x / size
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size


class InferenceLTMAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Similarity scores
        similarity_scores = k.mean(1)

        return x, similarity_scores


class InferenceLTMVisionTransformer(VisionTransformer):
    def __init__(self, tau=0.1, **kwargs):
        super().__init__(**kwargs)
        self.masks = []
        for block in self.blocks:
            block.merge_masker.tau = tau

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        s = torch.ones_like(x[..., 0, None])
        for block in self.blocks:
            x, s = block(x, s)
        x = self.norm(x)
        return x


# VIT
@register_model
def inference_ltm_vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=InferenceLTMBlock, **kwargs)
    model = create_vision_transformer(
        InferenceLTMVisionTransformer, "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def inference_ltm_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=InferenceLTMBlock, **kwargs)
    model = create_vision_transformer(
        InferenceLTMVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


# DEIT
@register_model
def inference_ltm_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=InferenceLTMBlock, **kwargs)
    model = create_vision_transformer(
        InferenceLTMVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def inference_ltm_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=InferenceLTMBlock, **kwargs)
    model = create_vision_transformer(
        InferenceLTMVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
