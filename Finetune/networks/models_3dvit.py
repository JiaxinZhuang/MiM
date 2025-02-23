"""Zhuang Jiaxin
lincolnz9511@gmail.com
Reference:
https://github.com/facebookresearch/mae/blob/main/models_vit.py
https://github.com/Project-MONAI/MONAI/blob/b61db797e2f3bceca5abbaed7b39bb505989104d/monai/networks/nets/vit.py
https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py
"""

from typing import Sequence, Union
from itertools import repeat
import collections.abc
from functools import partial
import torch.nn as nn
import torch

# timm==0.3.2
from timm.models.vision_transformer import Block

# from utils.misc import to_3tuple
# from utils.trace_utils import _assert


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # jia-xin
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # jia-xin
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.flatten = flatten
        # jiaxin
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # jiaxin
        B, C, D, H, W = x.shape
        _assert(D == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(H == self.img_size[1], f"Input image height ({H}) doesn't match model ({self.img_size[1]}).")
        _assert(W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer3D(nn.Module):
    """
    """
    def __init__(self,
                 in_channels: int,
                 img_size: Union[Sequence[int], int],
                 patch_size: Union[Sequence[int], int],
                 embed_dim: int = 1024, # hidden_size
                 # mlp_dim: int = 3072
                 mlp_ratio=4,
                 depth: int = 24, # num_layers
                 num_heads: int=16,
                 qkv_bias: bool=True,
                 norm_layer=nn.LayerNorm,
                 global_pool=False,
                 drop_rate: float=0,
                 classification: bool=False,
                 num_classes: int=0
                 ) -> None:

        super().__init__()

        # define structure
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        # if self.global_pool:
        #     norm_layer = kwargs['norm_layer']
        #     embed_dim = kwargs['embed_dim']
        #     self.fc_norm = norm_layer(embed_dim)

            # del self.norm

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.classification = classification
        if self.classification:
            self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        hidden_states_out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in [3, 6, 9]:
                hidden_states_out.append(x)

        if self.global_pool:
            # x = x[:, 1:, :].mean(dim=1)
            x = x.mean(dim=1)
            outcome = self.norm(x)
        else:
            x = self.norm(x)
            # outcome = x[:, 0]
            outcome = x

        if self.classification:
            # outcome = self.classification_head(x[:, 0])
            outcome = self.head(x)
            return outcome
        else:
            return outcome, hidden_states_out


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=1152, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=1344, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
