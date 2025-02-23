# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid depth, height and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print('embed_dim', embed_dim)
    print('grid_size', grid_size)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)
    grid = meshgrid2(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use third of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (D*H*W, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (D*H*W, D)
    return emb



def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return tuple(ans)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, new_size=None):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches

        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** (1/3.0))
        # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        new_size_ = []
        for ns, ps in zip(new_size, model.patch_embed.patch_size):
            new_size_.append(int(ns/ps))
        new_size = new_size_

        # class_token and dist_token are kept unchanged
        if orig_size != new_size[0] or orig_size != new_size[1] or orig_size != new_size[2]:
            print("Position interpolate from %dx%dx%d to %dx%dx%d" % (orig_size, orig_size, orig_size, new_size[0], new_size[1], new_size[2]))
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            extra_tokens = pos_embed_checkpoint[:, :1]
            # only the position tokens are interpolated
            # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_embed_checkpoint[:, 1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, orig_size, embedding_size).permute(0, 4, 1, 2, 3)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode='trilinear', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            print("Position interpolate done!")



# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# def interpolate_pos_embed(model, checkpoint_model):
    # if 'pos_embed' in checkpoint_model:
        # pos_embed_checkpoint = checkpoint_model['pos_embed']
        # embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = model.patch_embed.num_patches
        # num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        # if orig_size != new_size:
            # print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            # pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            # pos_tokens = torch.nn.functional.interpolate(
                # pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            # checkpoint_model['pos_embed'] = new_pos_embed
