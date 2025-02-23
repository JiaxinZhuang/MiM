"""Jiaxin ZHUANG @ Sep 12, 2023.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vision_transformer import PatchEmbed3d, Block, CBlock3d, Mlp
from utils.pos_embed import get_3d_sincos_pos_embed
# from utils.misc import print_with_timestamp


def byol_loss_fn(x, y):
    '''Compute the BYOL loss given two feature vectors.'''
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


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

        self.aggregate_mid = Mlp(in_features=decoder_embed_dim, hidden_features=decoder_embed_dim, act_layer=nn.GELU, drop=0)
        self.aggregate_down = Mlp(in_features=decoder_embed_dim, hidden_features=decoder_embed_dim, act_layer=nn.GELU, drop=0)

        self.aggregate_pred_up_feature = Mlp(in_features=decoder_embed_dim, hidden_features=decoder_embed_dim, act_layer=nn.GELU, drop=0)
        self.aggregate_pred_mid_feature = Mlp(in_features=decoder_embed_dim, hidden_features=decoder_embed_dim, act_layer=nn.GELU, drop=0)

        if self.args.attn_loss_name == 'byol':
            print('Using BYOL loss.')
            self.criterion = byol_loss_fn
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()

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

    def patchify(self, imgs, p=16):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L, patch_size**3 *1)
        """
        assert imgs.shape[3] == imgs.shape[4]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdz->nhwdpqzc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 1))
        return x

    def unpatchify(self, x, p=16):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, H, W, D)
        """
        # p = self.patch_embed3.patch_size[0]
        h = w = d = int(np.cbrt(x.shape[-2]))
        assert h * w * d == x.shape[-2]
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

    def random_masking_v2(self, x, mask_ratio, num_patches=None):
        '''Masking with different patch sizes.'''
        N = x.shape[0]

        LL = num_patches
        L = self.patch_embed4.num_patches

        len_keep = int(L * (1 - mask_ratio))

        # !Important
        noise = torch.rand(N, LL, device=x.device).repeat_interleave(dim=1, repeats=L//LL)

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

    def forward_encoder(self, x, mask_ratio, num_patches=None):
        '''
        x: [N, 1, H, W, D]
        '''
        # ! Important
        if num_patches is not None:
            ids_keep, mask, ids_restore = self.random_masking_v2(x, mask_ratio, num_patches=num_patches)
        else:
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
        stage1_embed = torch.gather(stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))
        stage3_embed = torch.gather(stage3_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage3_embed.shape[-1]))

        # apply Transformer blocks
        for blk in self.blocks4:
            if self.args.sr_ratio == 1:
                x = blk(x)
            else:
                x = blk(x, mask=mask_for_patch4, ids_restore=ids_restore, ids_keep=ids_keep)
        x = x + stage1_embed + stage2_embed + stage3_embed
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

        if self.args.not_use_attn:
            x_copy = None
        else:
            x_copy = x.clone()

        # predictor projection
        x = self.decoder_pred(x)

        return x, x_copy

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

    def select_patches(self, x, mask, patches_size, patches=None):
        '''Select patches based on the mask pattern
        x: [N, 1, 96x2, 96x2, 96x2]
        mask: [N, L]
        patches: patches size for each direction, e.g., xyz,
            for up: [2, 2, 2]
            for mid: [6, 6, 6]
        '''
        # [N, 1, 96x2, 96x2, 96x2] -> [N, 8, 96*96*96]
        out = self.patchify(x, p=patches_size[0])

        if patches is not None:
            new_mask = torch.nn.functional.max_pool1d(mask, kernel_size=27)

            N, L = new_mask.shape[0], new_mask.shape[1]
            noise = torch.rand(N, L, device=x.device)
            noise *= new_mask
            ids_keep = torch.argsort(noise, dim=1)[:, -self.args.sample_usual:]
        else:
            new_mask = mask
            N, L = new_mask.shape[0], new_mask.shape[1]
            noise = torch.rand(N, L, device=x.device)
            noise *= new_mask
            ids_keep = torch.argsort(noise, dim=1)[:, -self.args.sample_down:]

        # [N, L/2, 96*96*96]
        out = torch.gather(out, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, out.shape[-1]))
        # [N, L/2, 96*96*96] -> [N*L/2, 1, 96, 96, 96]

        if patches is not None:
            out = out.reshape(shape=(x.shape[0], -1, 96, 96, 96))
        else:
            out = out.reshape(shape=(x.shape[0], -1, 16, 16, 16))
        labels = ids_keep.to(torch.long).reshape(-1)
        return out, labels


    def forward(self, imgs):
        '''forward pass of the model
        Plan A [Current]:
            [N, 1, 96x2, 96x2, 96x2]
            All resize to [96, 96, 96] as the input
        Plan B:
            imgs: [N, 1, 96x4, 96x4, 96x2]
        '''
        N = imgs.shape[0]
        # if self.args.reconstruct_weight_usual == 1 and self.args.reconstruct_weight_usual == 0 and self.reconstruct_weight_down == 0.0:
        #         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask)
        # return loss, pred, mask

        # First, downsample to [96*2,96*2,96*2]
        patches_up = torch.tensor([2, 2, 2], device=imgs.device)
        # Resize from [96*2,96*2,96*2] to [96,96,96]
        img_up = F.interpolate(imgs, size=(self.args.up_roi_x, self.args.up_roi_y, self.args.roi_z), mode='trilinear', align_corners=False)
        # random masking and forward encoder
        # [N, L/2, D], [N, L], [N, L]
        latent_up, mask_up, ids_restore_up = self.forward_encoder(img_up, mask_ratio=self.args.mask_ratio_up, num_patches=torch.prod(patches_up))
        pred_up, pred_up_feature = self.forward_decoder(latent_up, ids_restore_up)  # [N, L, pux*puy*puz]
        loss_up = self.forward_loss(img_up, pred_up, mask_up)

        # Second as ususal, [96,96,96]
        # Select patches based on mask
        imgs_mid, labels_mid = self.select_patches(imgs, mask_up, patches_size=[96, 96, 96], patches=patches_up)
        imgs_mid_copy = imgs_mid.reshape(-1, 1, 96, 96, 96)
        latent_mid, mask_mid, ids_restore_mid = self.forward_encoder(imgs_mid_copy, mask_ratio=self.args.mask_ratio_mid)
        pred_mid, pred_mid_feature = self.forward_decoder(latent_mid, ids_restore_mid)  # [N, L, p*p*3]
        loss_mid = self.forward_loss(imgs_mid_copy, pred_mid, mask_mid)

        # Third resize to [96, 96, 96]
        imgs_down, labels_down = self.select_patches(imgs_mid_copy, mask_mid, patches_size=[16, 16, 16])
        imgs_down = imgs_down.reshape(-1, 1, 16, 16, 16)
        imgs_down = F.interpolate(imgs_down, size=(self.args.down_roi_x, self.args.down_roi_y, self.args.down_roi_z), mode='trilinear', align_corners=False)
        latent_down, mask_down, ids_restore_down = self.forward_encoder(imgs_down, mask_ratio=self.args.mask_ratio_down)
        pred_down, pred_down_feature = self.forward_decoder(latent_down, ids_restore_down)  # [N, L, p*p*3]
        loss_down = self.forward_loss(imgs_down, pred_down, mask_down)

        # Attention location back.
        # # Stage 2 back to stage 1
        if not self.args.not_use_attn:
            pred_mid_feature_copy = pred_mid_feature.clone()
            if self.args.atten_weight_uu != 0:
                # [N*sample_usual, L_u, e] -> [N*sample_usual, e] -> [N, sample_usual, e]
                pred_mid_feature = pred_mid_feature.mean(dim=1)
                embed_size = pred_mid_feature.shape[-1]
                mid_f = self.aggregate_mid(pred_mid_feature).reshape(N, -1, embed_size)
                # [N, L, e] -> [N, L//27, 27, e] -> [N, L//27, e]

                pred_up_feature = pred_up_feature.detach()
                L = pred_up_feature.shape[1]
                pred_up_feature = pred_up_feature.reshape(N, L//27, 27, -1)
                pred_up_feature = pred_up_feature.mean(dim=2)
                pred_up_feature = self.aggregate_pred_up_feature(pred_up_feature).reshape(N, -1, embed_size)

                # [N, sample_usual, e] mm [N, L//27, e] -> [N, sample_usual, L//27] -> [N*sample_usual, L//27]
                if self.args.attn_loss_name == 'byol':
                    # pred_up_feature = pred_up_feature[:, labels_mid, :]
                    # loss_upmid = self.criterion(pred_up_feature, mid_f)
                    #print(loss_upmid)
                               #down_f = down_f.reshape(-1, embed_size)
                    L = pred_up_feature.shape[0]
                    loss_upmid = 0
                    a_loss = 0
                    for i in range(L):
                        for j in range(mid_f.size(1)):
                            aa = pred_mid_feature[i]
                            bb = mid_f[i][j]
                            #print(aa.shape, bb.shape)
                            a_loss += self.criterion(aa, bb)
                            #print(a_loss)
                    loss_upmid = a_loss / (L * mid_f.size(1))
                else:
                    #mid_up_logits = self.criterion(mid_f, pred_up_feature)
                    mid_f = F.normalize(mid_f, dim=-1, p=2)
                    pred_up_feature = F.normalize(pred_up_feature, dim=-1, p=2)
                    mid_up_logits = torch.einsum('ijk,imk->ijm', mid_f, pred_up_feature)
                    mid_up_logits = mid_up_logits.reshape(-1, L//27)
                    loss_upmid = self.criterion(mid_up_logits, labels_mid)
            else:
                loss_upmid = torch.tensor(0.0).cuda()

            if self.args.atten_weight_ud != 0:
                # Stage 3 back to stage 2
                # [N*sample_down*sample_usual, L_d, e] -> [N*sample_usual, e] -> [N, sample_usual, e]
                pred_down_feature = pred_down_feature.mean(dim=1)
                embed_size = pred_down_feature.shape[-1]
                down_f = self.aggregate_down(pred_down_feature).reshape(N*self.args.sample_down, -1, embed_size)

                pred_mid_feature_copy = pred_mid_feature_copy.detach()
                pred_mid_feature_copy = pred_mid_feature_copy.reshape(-1, 216, embed_size)
                pred_mid_feature_copy = pred_mid_feature_copy.mean(dim=1)
                NN = pred_mid_feature_copy.shape[0]
                pred_mid_feature_copy = self.aggregate_pred_mid_feature(pred_mid_feature_copy).reshape(NN, -1, embed_size)
                #.reshape(N, -1, embed_size)

                if self.args.attn_loss_name == 'byol':
                    #print(pred_mid_feature.shape)
                    #print(down_f.shape)
                    # print(labels_down)
                    # labels_down = labels_down.reshape(N, -1)
                    # pred_mid_feature = pred_mid_feature[:, labels_down, :]
                    # pred_mid_feature = pred_mid_feature[labels_down, :]
                    #down_f = down_f.reshape(-1, embed_size)
                    L = pred_mid_feature_copy.shape[0]
                    loss_middown = 0
                    a_loss = 0
                    for i in range(L):
                        for j in range(down_f.size(1)):
                            aa = pred_mid_feature_copy[i]
                            bb = down_f[i][j]
                            #print(aa.shape, bb.shape)
                            a_loss += self.criterion(aa, bb)
                            #print(a_loss)
                    loss_middown = a_loss / (L * down_f.size(1))
                else:
                    down_f = F.normalize(down_f, dim=-1, p=2)
                    pred_mid_feature_copy = F.normalize(pred_mid_feature_copy, dim=-1, p=2)
                    #TODO
                    print(down_f.shape, pred_mid_feature_copy.shape)
                    down_mid_logits = torch.einsum('ijk,imk->ijm', down_f, pred_mid_feature_copy)
                    L = pred_mid_feature_copy.shape[1]
                    down_mid_logits = down_mid_logits.reshape(-1, L)
                    loss_middown = self.criterion(down_mid_logits, labels_down)
            else:
                loss_middown = torch.tensor(0.0).cuda()
        else:
            loss_upmid = loss_middown = torch.tensor(0.0).cuda()
        return [loss_up, loss_mid, loss_down, loss_upmid, loss_middown], pred_up, pred_mid, pred_down, mask_up, mask_mid, mask_down, ids_restore_up, ids_restore_mid, ids_restore_down, [img_up, imgs_mid_copy, imgs_down]


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


# def convmae_convvit_3d_base_patch16_dec512d8b(**kwargs):
#     ''' 3d convvit with input size
#     :96->48->24->12
#     '''
#     model = MaskedAutoencoderConvViT3d(
#         in_chans=1,
#         img_size=[[96, 96, 96], [48, 48, 48], [24, 24, 24], [12, 12, 12]],
#         patch_size=[2, 2, 2, 2],
#         embed_dim=[48, 96, 192, 384],
#         depth=[2, 2, 2, 11],
#         num_heads=12,
#         decoder_embed_dim=576,
#         decoder_depth=8,
#         decoder_num_heads=16, mlp_ratio=[4, 4, 4, 4],
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model    


def convmae_convvit_3d_base_patch16_i128_dec512d8b(**kwargs):
    ''' 3d convvit with input size 128x32x16'''
    model = MaskedAutoencoderConvViT3d(
        img_size=[128, 32, 16],
        in_chans=1,
        patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11],
        num_heads=12, decoder_embed_dim=576, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#TODO
def convmae_convvit_3d_tiny_patch16_dec512d8b(**kwargs):
    ''' 3d convvit with input size
    :96->48->24->12
    '''
    model = MaskedAutoencoderConvViT3d(
        in_chans=1,
        img_size=[[96, 96, 96], [48, 48, 48], [24, 24, 24], [12, 12, 12]],
        patch_size=[2, 2, 2, 2],
        # embed_dim=[48, 96, 192, 384],
        embed_dim=[12, 24, 48, 96],
        depth=[2, 2, 2, 11],
        num_heads=3, # Change
        decoder_embed_dim=576,
        decoder_depth=8,
        decoder_num_heads=16, mlp_ratio=[4, 4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
convmae_convvit_tiny_patch16 = convmae_convvit_3d_tiny_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
convmae_convvit_base_patch16 = convmae_convvit_3d_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
convmae_convvit_base_patch16_i128 = convmae_convvit_3d_base_patch16_i128_dec512d8b  # decoder: 512 dim, 8 blocks

#convmae_convvit_base_patch16 = convmae_convvit_3d_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
