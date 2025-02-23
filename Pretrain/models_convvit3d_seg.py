'''Jiaxin ZHUANG, @Sep 12, 2023.
Ref: https://github.com/Alpha-VL/ConvMAE/blob/main/SEG/backbone/convmae.py
     https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html
     https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
Designed for convmae_v2.

'''

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_
from monai.utils import optional_import

from vision_transformer import PatchEmbed3d, CBlock3d, Mlp, MulAttention, RelativePositionBias3d
# , Block, CBlock3d
rearrange, _ = optional_import("einops", name="rearrange")



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        '''x: B, C, D, H, W'''
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'


class Attention(nn.Module):
    '''Multi-head self-attention with relative positional encoding.'''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, 2*Wd-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords_d = torch.arange(window_size[2])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Wh, Ww, Wd
            coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 2] += window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * window_size[2] - 1

            relative_position_index = relative_coords.sum(-1)
            # relative_position_index = \
            #     torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            # relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # relative_position_index[0, 0:] = self.num_relative_distance - 3
            # relative_position_index[0:, 0] = self.num_relative_distance - 2
            # relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] * self.window_size[2],
                    self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):
    '''Block for transformer with *window* attention mechanism.
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, window_size_ds=None, attn_head_dim=None, args=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.attn = Attention(
        #    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #    attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        if args and args.sr_ratio != 1:
            print('MulAtt')
            self.attn = MulAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                sr_ratio=args.sr_ratio, window_size_1=window_size_ds, window_size_2=window_size)
        else:
            print('Att')
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class ConvViT3d(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=[512, 128, 64], patch_size=[4,2,2], in_chans=3, num_classes=80, embed_dim=[256,384,768], depth=[2,2,11],
                 num_heads=12, mlp_ratio=[4,4,4], qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 init_values=1, use_checkpoint=False,
                 use_abs_pos_emb=True, use_rel_pos_bias=True, use_shared_rel_pos_bias=False,
                 out_indices=[3, 5, 7, 11], fpn1_norm='SyncBN', args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # if hybrid_backbone is not None:
            # self.patch_embed = HybridEmbed(
                # hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        # else:
        self.patch_embed1 = PatchEmbed3d(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed3d(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed3d(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed3d(
            img_size=img_size[3], patch_size=patch_size[3], in_chans=embed_dim[2], embed_dim=embed_dim[3])
        # Before transformer block.
        self.patch_embed5 = nn.Linear(embed_dim[3], embed_dim[3])

        num_patches = self.patch_embed4.num_patches

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[3]))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias3d(window_size=self.patch_embed4.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_rel_pos_bias = use_rel_pos_bias
        # self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        # stage 3
        self.blocks3 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        # stage 4
        self.blocks4 = nn.ModuleList([
            Block(
                dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed4.patch_shape if use_rel_pos_bias else None, window_size_ds=[ps//2 for ps in self.patch_embed4.patch_shape] if args.sr_ratio != 1 else None, args=args)
            for i in range(depth[3])])

        #self.fpn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fpn = nn.Sequential(
            nn.Conv3d(embed_dim[3], embed_dim[-1], kernel_size=2, stride=2),
            # nn.Conv3d(embed_dim[3], embed_dim[-1], kernel_size=3, stride=2, padding=1),
            nn.SyncBatchNorm(embed_dim[-1]),
            nn.GELU(),
            )
        # self.norm = norm_layer(embed_dim[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        '''Fix the init weight of the model.'''
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks4):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward_features(self, x, normalize=False):
        '''Forward function.'''
        features = []
        B = x.shape[0]

        # stage 1
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        features.append(self.proj_out(x, normalize))

        # stage 2
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        # features.append(x)
        features.append(self.proj_out(x, normalize))

        # stage 3
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        # features.append(x)
        features.append(self.proj_out(x, normalize))

        # stage 4
        x = self.patch_embed4(x)
        Hp, Wp, Dp = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed5(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks4:
            # if self.use_checkpoint:
                # x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            # else:
            x = blk(x, rel_pos_bias)
        # x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp, Dp)
        # features.append(x.contiguous())
        features.append(self.proj_out(x.contiguous(), normalize))

        # stage 5, downsample
        x = self.fpn(x)
        features.append(self.proj_out(x, normalize))

        return tuple(features)

    def forward(self, x, normalize=False):
        '''Forward function.'''
        x = self.forward_features(x, normalize)
        return x


def convvit3d_base_patch16_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    :96->48->24->12
    '''
    model = ConvViT3d(
        in_chans=1,
        img_size=[[96, 96, 96], [48, 48, 48], [24, 24, 24], [12, 12, 12]],
        patch_size=[2, 2, 2, 2],
        embed_dim=[48, 96, 192, 384, 768],
        depth=[2, 2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit3d_tiny_patch16_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    :96->48->24->12
    '''
    model = ConvViT3d(
        in_chans=1,
        img_size=[[96, 96, 96], [48, 48, 48], [24, 24, 24], [12, 12, 12]],
        patch_size=[2, 2, 2, 2],
        # embed_dim=[48, 96, 192, 384, 768],
        embed_dim=[12, 24, 48, 96, 192],
        depth=[2, 2, 2, 11],
        # num_heads=12,
        num_heads=3,
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def convvit3d_base_patch16_192x192x32_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    :192->96->48->24
    '''
    model = ConvViT3d(
        in_chans=1,
        img_size=[[192, 192, 32], [96, 96, 16], [48, 48, 8], [24, 24, 4]],
        #img_size=[[192, 192, 16], [96, 96, 8], [48, 48, 4], [24, 24, 2]],
        patch_size=[2, 2, 2, 2],
        embed_dim=[48, 96, 192, 384, 768],
        depth=[2, 2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit3d_base_patch16_128x128x128_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    '''
    model = ConvViT3d(
        in_chans=4,
        img_size=[[128, 128, 128], [64, 64, 64], [32, 32, 32], [16, 16, 16]],
        patch_size=[2, 2, 2, 2],
        embed_dim=[48, 96, 192, 384, 768],
        depth=[2, 2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit3d_base_patch16_224x224x128_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    '''
    model = ConvViT3d(
        in_chans=4,
        img_size=[[224, 224, 128], [112, 112, 64], [56, 56, 32], [28, 28, 16]],
        patch_size=[2, 2, 2, 2],
        embed_dim=[48, 96, 192, 384, 768],
        depth=[2, 2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



convvit3d_tiny_patch16 = convvit3d_tiny_patch16_dec512d8b
convvit3d_base_patch16 = convvit3d_base_patch16_dec512d8b
convvit3d_base_patch16_im = convvit3d_base_patch16_192x192x32_dec512d8b
convvit3d_base_patch16_mri = convvit3d_base_patch16_128x128x128_dec512d8b
convvit3d_base_patch16_mri_v2 = convvit3d_base_patch16_224x224x128_dec512d8b
