"""Jiaxin ZHUANG @ Sep 12, 2023.
"""

from functools import partial
import torch
import torch.nn as nn
import numpy as np

from vision_transformer import PatchEmbed3d, Block, CBlock3d

from utils.MAE.pos_embed import get_3d_sincos_pos_embed


class MaskedAutoencoderConvViT3d(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()
        self.args = args
        # --------------------------------------------------------------------------
        # ConvMAE3d encoder specifics
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
        self.stage1_output_decode = nn.Conv3d(embed_dim[0], embed_dim[3], 2*2*2, stride=2*2*2)
        self.stage2_output_decode = nn.Conv3d(embed_dim[1], embed_dim[3], 2*2, stride=2*2)
        self.stage3_output_decode = nn.Conv3d(embed_dim[2], embed_dim[3], 2, stride=2)

        num_patches = self.patch_embed4.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[3]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            CBlock3d(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2],
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            Block(
                dim=embed_dim[3], num_heads=num_heads, mlp_ratio=mlp_ratio[3],
                qkv_bias=True, qk_scale=None, norm_layer=norm_layer, args=args)
            for i in range(depth[3])])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3])**3 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        '''Initialize model weights.'''
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(np.cbrt(self.patch_embed4.num_patches)), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(np.cbrt(self.patch_embed4.num_patches)), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L, patch_size**3 *1)
        """
        p = 16
        assert imgs.shape[3] == imgs.shape[4]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdz->nhwdpqzc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, H, W, D)
        """
        # p = self.patch_embed3.patch_size[0]
        p = 16
        h = w = d = int(np.cbrt(x.shape[1]))
        assert h * w * d == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
        x = torch.einsum('nhwdpqzc->nchpwqdz', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p, d*p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], batch_size, sequence, dimension.
        """
        N = x.shape[0]
        L = self.patch_embed4.num_patches
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unsuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        '''
        x: [N, 1, H, W, D]
        '''
        # embed patches
        ids_keep, mask, ids_restore = self.random_masking(x, mask_ratio)


        # mask for stage 1, 2 and stage 3.
        mask_for_patch1 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 8**3).reshape(-1, 6, 6, 6, 8, 8, 8).permute(0, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 48, 48, 48).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 4**3).reshape(-1, 6, 6, 6, 4, 4, 4).permute(0, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 24, 24, 24).unsqueeze(1)
        mask_for_patch3 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 2**3).reshape(-1, 6, 6, 6, 2, 2, 2).permute(0, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 12, 12, 12).unsqueeze(1)
        mask_for_patch4 = mask.reshape(-1, 6, 6, 6).unsqueeze(-1).repeat(1, 1, 1, 1, 1**1).reshape(-1, 6, 6, 6, 1, 1, 1).permute(0, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 6, 6, 6).unsqueeze(1)

        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x, 1 - mask_for_patch3)
        stage3_embed = self.stage3_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed4(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed5(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        # stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        # stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
        # stage3_embed = torch.gather(stage3_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage3_embed.shape[-1]))

        # apply Transformer blocks
        for blk in self.blocks4:
            if self.args.sr_ratio == 1:
                x = blk(x)
            else:
                x = blk(x, mask=mask_for_patch4, ids_restore=ids_restore, ids_keep=ids_keep)
        # x = x + stage1_embed + stage2_embed + stage3_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''forward decoder to reconstruct'''
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]  - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        '''forward pass of the model'''
        latent, mask, ids_restore = self.forward_encoder(imgs, self.args.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, ids_restore


# def convmae_convvit_base_patch16_dec512d8b(**kwargs):
    # model = MaskedAutoencoderConvViT(
        # img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        # decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        # mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # return model

def convmae_convvit_3d_base_patch16_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    :96->48->24->12
    '''
    model = MaskedAutoencoderConvViT3d(
        in_chans=1,
        #img_size=[96, 48, 24, 12],
        img_size=[[96, 96, 96], [48, 48, 48], [24, 24, 24], [12, 12, 12]],
        patch_size=[2, 2, 2, 2],
        embed_dim=[48, 96, 192, 384],
        depth=[2, 2, 2, 11],
        num_heads=12,
        decoder_embed_dim=576,
        decoder_depth=8,
        decoder_num_heads=16, mlp_ratio=[4, 4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def convmae_convvit_3d_base_patch16_i128_dec512d8b(**kwargs):
    ''' 3d convvit with input size 128x32x16'''
    model = MaskedAutoencoderConvViT3d(
        img_size=[128, 32, 16],
        in_chans=1,
        patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11],
        num_heads=12, decoder_embed_dim=576, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
convmae_convvit_base_patch16 = convmae_convvit_3d_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
convmae_convvit_base_patch16_i128 = convmae_convvit_3d_base_patch16_i128_dec512d8b  # decoder: 512 dim, 8 blocks
