import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, Block, LayerScale, VisionTransformer

from ltmp.token_merging import bipartite_soft_matching, merge_wavg

from .mergeprune import parse_r
from .utils import create_vision_transformer


class PMBlock(Block):
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
        self.attn = PMAttention(
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

        self.r_prune = 0
        self.r_merge = 0

    def forward(self, x, size):
        x_attn, metric, sorted_ranking = self.attn(self.norm1(x), size)
        x = x + self.drop_path1(self.ls1((x_attn)))

        r = min(self.r_prune, (x.shape[1] - 1) // 2)
        min_r = -r if r != 0 else None
        x = x.gather(dim=1, index=sorted_ranking.expand(x.shape))[:, :min_r, ...]

        metric = metric.gather(dim=1, index=sorted_ranking.expand(metric.shape))[:, :min_r, ...]

        size = size.gather(dim=1, index=sorted_ranking.expand(size.shape))[:, :min_r, ...]

        if self.r_merge > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                self.r_merge,
                True,
                False,
            )
            x, size = merge_wavg(merge, x, size)

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size


class PMAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply proportional attention
        attn = attn + size.log()[:, None, None, :, 0]

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

        # Calculate similarity scores
        similarity_scores = k.mean(1)

        return x, similarity_scores, importance_scores_indices


class PMVisionTransformer(VisionTransformer):
    def __init__(self, r=0, **kwargs):
        super().__init__(**kwargs)
        merge_r, prune_r = parse_r(len(self.blocks), r)
        for block in self.blocks:
            block.r_merge = merge_r.pop(0)
            block.r_prune = prune_r.pop(0)

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
def pm_vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=PMBlock, **kwargs)
    model = create_vision_transformer(
        PMVisionTransformer, "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def pm_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=PMBlock, **kwargs)
    model = create_vision_transformer(
        PMVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


# DEIT
@register_model
def pm_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=PMBlock, **kwargs)
    model = create_vision_transformer(
        PMVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def pm_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=PMBlock, **kwargs)
    model = create_vision_transformer(
        PMVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
