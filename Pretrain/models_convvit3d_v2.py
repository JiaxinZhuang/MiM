'''Jiaxin ZHUNAG @Sep 12, 2023.
'''

from functools import partial

import torch
import torch.nn as nn
import pdb
import vision_transformer


class ConvViT3d(vision_transformer.ConvViT3d):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(ConvViT3d, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim[-1])

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        x = self.patch_embed4(x)
        x = x.flatten(2).permute(0, 2, 1)

        x = self.patch_embed5(x)
        x = x + self.pos_embed
        for blk in self.blocks4:
            x = blk(x)

        if self.global_pool:
            x = x[:, :, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x
        return outcome


def convvit_3d_base_patch16(**kwargs):
    model = ConvViT3d(
        in_chans=1,
        img_size=[96, 48, 24, 12], 
        patch_size=[2, 2, 2, 2], 
        embed_dim=[48, 96, 192, 384], 
        depth=[2, 2, 2, 11], 
        num_heads=12, mlp_ratio=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit_3d_base_patch16_i128(**kwargs):
    model = ConvViT3d(
        in_chans=1,
        img_size=[128, 32, 16], 
        patch_size=[4, 2, 2], embed_dim=[256, 384, 768], 
        depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model