import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp, PatchDropout, PatchEmbed
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, Block, LayerScale, VisionTransformer

from ltmp.utils import parse_r

from .utils import create_vision_transformer


class MBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MAttention(
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

    def forward(self, x: torch.Tensor, size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_attn, metric = self.attn(self.norm1(x), size)
        x = x + self.drop_path1(self.ls1((x_attn)))

        n, t, c = x.shape
        r = min(self.r, (t - 1) // 2)
        metric = metric / metric.norm(p=2, dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        x = x * size
        src_x, dst_x = x[..., ::2, :], x[..., 1::2, :]
        src_s, dst_s = size[..., ::2, :], size[..., 1::2, :]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        unm_idx, _ = unm_idx.sort(dim=1)
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        _, t1, _ = src_x.shape

        unm_x = src_x.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        unm_s = src_s.gather(dim=-2, index=unm_idx.expand(n, t1 - r, 1))
        src_x = src_x.gather(dim=-2, index=src_idx.expand(n, r, c))
        src_s = src_s.gather(dim=-2, index=src_idx.expand(n, r, 1))
        dst_x = dst_x.scatter_reduce(-2, dst_idx.expand(n, r, c), src_x, reduce="sum")
        dst_s = dst_s.scatter_reduce(-2, dst_idx.expand(n, r, 1), src_s, reduce="sum")
        x = torch.cat([unm_x, dst_x], dim=1)
        size = torch.cat([unm_s, dst_s], dim=1)
        x = x / size

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size


class MAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Return k as well here
        return x, k.mean(1)


class MVisionTransformer(VisionTransformer):
    # We need to copy most of timm VisionTransfomer constructor
    # to change blocks to an nn.ModuleList so it can be used with JIT
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        mlp_layer=Mlp,
        r=0,
    ):
        super(VisionTransformer, self).__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)
        r = parse_r(len(self.blocks), r)
        for block in self.blocks:
            block.r = r.pop(0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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
def m_vit_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "vit_tiny_patch16_224", pretrained=pretrained, **dict(model_kwargs, **kwargs)
    )
    return model


@register_model
def m_vit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def m_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(MVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
    return model


@register_model
def m_vit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


# DEIT
@register_model
def m_deit_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "deit_tiny_patch16_224", pretrained=pretrained, **dict(model_kwargs, **kwargs)
    )
    return model


@register_model
def m_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def m_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=MBlock, **kwargs)
    model = create_vision_transformer(
        MVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
