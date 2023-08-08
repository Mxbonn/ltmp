import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, Block, LayerScale, VisionTransformer

from ltmp.utils import parse_r

from .utils import create_vision_transformer


class PBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.r = 0
        self.trace_source = False
        self.source = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, sorted_ranking = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1((x_attn)))

        r = min(self.r, (x.shape[1] - 1) // 2)
        min_r = -r if r != 0 else None
        x = x.gather(dim=1, index=sorted_ranking.expand(x.shape))[:, :min_r, ...]
        if self.trace_source:
            if self.source is None:
                self.source = [sorted_ranking[:, :min_r, :]]
            else:
                self.source.append(self.source[-1].gather(dim=1, index=sorted_ranking)[:, :min_r, :])

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PAttention(Attention):
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Calculate importance scores and indices
        importance_scores = torch.mean(attn, dim=(1, 2))  # mean attention
        importance_scores[..., 0] = math.inf
        _, importance_scores_indices = importance_scores.sort(dim=-1, descending=True)
        importance_scores_indices = importance_scores_indices[..., None]

        return x, importance_scores_indices


class PVisionTransformer(VisionTransformer):
    def __init__(self, r=0, **kwargs):
        super().__init__(**kwargs)
        r = parse_r(len(self.blocks), r)
        for block in self.blocks:
            block.r = r.pop(0)


# VIT
@register_model
def p_vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=PBlock, **kwargs)
    model = create_vision_transformer(
        PVisionTransformer, "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def p_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=PBlock, **kwargs)
    model = create_vision_transformer(PVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
    return model


# DEIT
@register_model
def p_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=PBlock, **kwargs)
    model = create_vision_transformer(
        PVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def p_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=PBlock, **kwargs)
    model = create_vision_transformer(
        PVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
