import torch
import torch.nn as nn
import math
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import pdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class CMlp3d(nn.Module):
    ''' Convolutional MLP 3d version
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
        x: [B, embed_dim, H, W, D]
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CBlock3d(nn.Module):
    ''' Convolutional Block for Convolutional ViT, 3d version
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    '''
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp3d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        '''
        x: [B, embed_dim, H, W, D]
        '''
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if args and args.sr_ratio != 1:
            # print('MulAtt')
            self.attn = MulAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                sr_ratio=args.sr_ratio)
        else:
            # print('Att')
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), **kwargs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3d(nn.Module):
    """ Image to Patch Embedding 3d version
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert len(img_size) == 3
        # img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = (img_size[0] // patch_size[0], 
                            img_size[1] // patch_size[1],
                            img_size[2] // patch_size[2])

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W, D = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2] \
        #    and f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.act(x)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvViT3d(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed1 = PatchEmbed3d(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed3d(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed3d(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        num_patches = self.patch_embed3.num_patches
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])


        self.norm = norm_layer(embed_dim[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed
        for blk in self.blocks3:
            x = blk(x)
        x = self.norm(x)
        return x.mean(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


        
class MulAttention(nn.Module):
    ''' Multi-head Self-attention using mutli-scale features, default sr rario is 2.
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2, window_size_1=None, window_size_2=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.max_pooling = nn.MaxPool3d(kernel_size=2, stride=2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            #if sr_ratio==8:
            #    self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
            #    self.norm1 = nn.LayerNorm(dim)
            #    self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
            #    self.norm2 = nn.LayerNorm(dim)
            #if sr_ratio==4:
            #    self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
            #    self.norm1 = nn.LayerNorm(dim)
            #    self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
            #    self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv3d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv3d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv3d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv3d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        if window_size_1 is not None:
            self.ws1 = RelativePositionBias3d(window_size=window_size_1, num_heads=num_heads//2)            
        else:
            self.ws1 = None

        if window_size_2 is not None:
            self.ws2 = RelativePositionBias3d(window_size=window_size_2, num_heads=num_heads//2)
        else:
            self.ws2 = None
                    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=6, W=6, D=6, mask=None, ids_restore=None, ids_keep=None, rel_pos_bias=None):
        '''
        mask: [B, C, H, W, D]
        '''
        #TODO H, W, D should be calculated automatically
        #print('bb', x.shape)
        B, N, C = x.shape
        # [B, N, nh, C//nh] => [B, nh, N, C//nh]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            if mask is not None and ids_restore is not None:
                # append mask tokens and unshuffle
                mask_tokens = torch.zeros(B, ids_restore.shape[1] - N, C, device=x.device)
                # B, N, C => B, N + M, C
                x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
                x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))  # unshuffle

                x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
                # [B, C, H, W, D] => [B, C, H//sr, W//sr, D//sr] => [B, C, N]

                x_out_1 = self.sr1(x_)
                #print(x_out_1.shape)
                #x_out_1_mask = self.max_pooling(1 - mask)
                #print('x_out_1_mask', x_out_1_mask.shape)
                # [B, 1, H, W, D] -> [B, 1, H/2, W/2, D/2] -> [B, 1, HWD/8] -> [B, HWD/8, 1]
                #mask_1 = self.max_pooling(1-mask).flatten(2).permute(0, 2, 1)
                #print('mask_1', mask_1.shape)
                #ids_keep_1 = torch.nonzero()
                #ids_keep_1 = torch.nonzero(self.max_pooling(1-mask))
                #print('ids_keep_1', ids_keep_1.shape)
                #print(x_out_1.shape)
                #x_out_1 = torch.gather(x_out_1, dim=1, index=ids_keep_1.unsqueeze(-1).repeat(1, 1, x_out_1.shape[-1]))
                #x_out_1 = x_out_1 * x_out_1_mask
                #x_out_1 = torch.gather(x_out_1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_out_1.shape[-1]))
                x_1 = self.act(self.norm1(x_out_1.reshape(B, C, -1).permute(0, 2, 1)))

                # [B, C, H, W, D] => [B, C, H//sr, W//sr, D//sr] => [B, C, N]
                x_out_2 = self.sr2(x_)
                #print('x_out_2', x_out_2.shape)
                #x_out_2_mask = self.max_pooling(1 - mask)
                #print('x_out_2_mask', x_out_2_mask.shape)
                #x_out_2 = x_out_2 * x_out_2_mask
                #x_out_2 = torch.gather(x_out_2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_out_2.shape[-1]))
                #x_out_2 = torch.gather(x_out_1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_out_1.shape[-1]))
                x_2 = self.act(self.norm2(x_out_2.reshape(B, C, -1).permute(0, 2, 1)))
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
                x_out_1 = self.sr1(x_)
                x_1 = self.act(self.norm1(x_out_1.reshape(B, C, -1).permute(0, 2, 1)))
                #print(x_1.shape)
                x_out_2 = self.sr2(x_)
                x_2 = self.act(self.norm2(x_out_2.reshape(B, C, -1).permute(0, 2, 1)))
                #print('x_2', x_2.shape)

            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1] #B head N C
            k2, v2 = kv2[0], kv2[1]

            attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale

            #TODO, still need to udnerstand and deal
            #if self.ws1 is not None:
            #    #attn1 torch.Size([4, 6, 216, 27])
            #    print('attn1', attn1.shape)
            #    attn1 = attn1 + self.ws1()
            #    print('mul atten with rel', attn1.shape)

            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            
            #TODO
            #if mask is not None and ids_restore is not None:
            #    # v1: [B, N, C//2]
            #    v1_tmp = v1.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2)
            #    mask_tokens = torch.zeros(B, ids_restore.shape[1] - N, C//2, device=v1_tmp.device)
            #    v1_tmp_ = torch.cat([v1_tmp, mask_tokens], dim=1)  # no cls token
            #    v1_tmp_ = torch.gather(v1_tmp_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C//2))  # unshuffle
            #    v1_tmp_ = v1_tmp.view(B,C//2, H//self.sr_ratio, W//self.sr_ratio, D//self.sr_ratio)
            #    v1_out = self.local_conv1(v1_tmp_)
            #    ids_keep_1 = torch.nonzero(self.max_pooling(1-mask))
            #    v_out_1 = torch.gather(v1_out, dim=1, index=ids_keep_1.unsqueeze(-1).repeat(1, 1, v1_tmp.shape[-1]))
            #    v1 = v1 + v_out_1
            #    #view(B,C//2, H//self.sr_ratio, W//self.sr_ratio, D//self.sr_ratio)
            #else:
            #v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2).\
            #    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1)).\
            #        view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio, D//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            #print('v1', v1.shape)

            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)

            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale

            if self.ws2 is not None:
                #print('attn2', attn2.shape)
                #print(self.ws2().shape)
                attn2 = attn2 + self.ws2()
                #print('mul atten with rel', attn2.shape)

            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            #TODO
            #if mask and ids_restore:
            #    v2_tmp = v2.transpose(1, 2).reshape(B, -1, C//2).transpose(1, 2)
            #    mask_tokens = torch.zeros(B, ids_restore.shape[1] - N, C//2, device=v2_tmp.device)
            #    v2_tmp_ = torch.cat([v2_tmp, mask_tokens], dim=1)
            #    v2_tmp_ = torch.gather(v2_tmp_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C//2))
            #    v2_tmp_ = v2_tmp.view(B,C//2, H*2//self.sr_ratio, W*2//self.sr_ratio, D*2//self.sr_ratio)
            #    v2_out = self.local_conv2(v2_tmp_)
            #    ids_keep_2 = torch.nonzero(self.max_pooling(1-mask))
            #    v_out_2 = torch.gather(v2_out, dim=1, index=ids_keep_2.unsqueeze(-1).repeat(1, 1, v2_tmp.shape[-1]))
            #    v2 = v2 + v_out_2
            #else:
            #    v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
            #                            transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio, D*2//self.sr_ratio)).\
            #        view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio, D*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

            x = torch.cat([x1,x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W, D)).view(B, C, N).transpose(1, 2)
        #print(x.shape)
        x = self.proj(x)
        #print(x.shape)
        x = self.proj_drop(x)

        return x

        
class RelativePositionBias3d(nn.Module):
    '''Relative Position Bias in 3d
    Ref: https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html
    '''
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1), num_heads
            )
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        # self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) 
        # self.relative_position_bias_table = nn.Parameter(
            # torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(window_size[0])
        # coords_w = torch.arange(window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # relative_position_index = \
        #     torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        # relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # relative_position_index[0, 0:] = self.num_relative_distance - 3
        # relative_position_index[0:, 0] = self.num_relative_distance - 2
        # relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wd,Wh*Ww*Wd,nH
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # nH, Wh*Ww*Wd, Wh*Ww*Wd