# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.cls_head = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_repl_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_no_pos = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_ha_pos = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])


        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.cls_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.cls_pred = nn.Linear(decoder_embed_dim, patch_size**2, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_repl_pos, std=.02)
        torch.nn.init.normal_(self.decoder_no_pos, std=.02)
        torch.nn.init.normal_(self.decoder_ha_pos, std=.02)
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio,pos_embed):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_shuffle = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        pos_embed=pos_embed.expand(N,-1,-1)
        pos_embed_shuffle= torch.gather(pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_shuffle[:,:len_keep,:] = x_shuffle[:,:len_keep,:]+pos_embed_shuffle
        repel_pos = self.encoder_repl_pos.repeat(N, L-len_keep , 1)
        x_shuffle[:,len_keep:,:] = x_shuffle[:,len_keep:,:]+repel_pos

        return x_shuffle,ids_shuffle,len_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        x, ids_shuffle,len_keep = self.random_masking(x, mask_ratio,self.pos_embed[:, 1:, :])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_shuffle, len_keep

    def forward_decoder(self, x, ids_shuffle, len_keep):
        # embed tokens
        x = self.decoder_embed(x)
        N,L,D = x.size()
        decoder_pos_embed=self.decoder_pos_embed[:,1:,:]
        decoder_pos_embed=decoder_pos_embed.expand(N,-1,-1)
        # decoder_pos_embed=torch.gather(decoder_pos_embed, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        noise=torch.rand(N,L-len_keep-1, device=x.device)#减去cls tokens
        argsort=torch.argsort(noise,dim=1)
        decoder_pos_shuffle=torch.gather(ids_shuffle[:,len_keep:], dim=1, index=argsort)
        decoder_pos_shuffle=torch.cat([ids_shuffle[:,:len_keep],decoder_pos_shuffle],dim=1)
        decoder_pos_embed=torch.gather(decoder_pos_embed, dim=1, index=decoder_pos_shuffle.unsqueeze(-1).repeat(1, 1, D))
        decoder_pos_embed=torch.cat([self.decoder_pos_embed[:,:1,:].expand(N,-1,-1),decoder_pos_embed],dim=1)
        x=x+decoder_pos_embed
        no_pos=self.decoder_no_pos.repeat(N,L-len_keep-1,1)
        ha_pos=self.decoder_ha_pos.repeat(N,len_keep,1)
        wo_pos=torch.cat([ha_pos,no_pos],dim=1)
        x[:,1:,:]=x[:,1:,:]+wo_pos

        # ids_shuffle[:,len_keep:]=argsort

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x , ids_shuffle ,decoder_pos_shuffle

    def forward_classifaction_loss(self, x , ids_shuffle):
        # embed tokens
        x = self.cls_head(x)
        x = self.cls_norm(x)
        # predictor projection
        x = self.cls_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        y_label=ids_shuffle.reshape(-1).long()
        N,L,D=x.size()
        y_hat=x.reshape(-1,D)

        cross_func=nn.CrossEntropyLoss()
        loss=cross_func(y_hat,y_label)
        return loss

    def forward_loss(self, imgs, pred, shuffle):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        N,L,D = target.size()
        target=torch.gather(target, dim=1, index=shuffle.unsqueeze(-1).repeat(1, 1, D))
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch

        return loss

    def forward(self, aug_samples,imgs, mask_ratio=0.75):
        _lambda=0.1
        latent, ids_shuffle, len_keep = self.forward_encoder(aug_samples, mask_ratio)
        pred, ids_shuffle, decoder_pos_shuffle = self.forward_decoder(latent, ids_shuffle ,len_keep)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, decoder_pos_shuffle) + _lambda * self.forward_classifaction_loss(latent , ids_shuffle)
        # loss = self.forward_loss(imgs, pred, decoder_pos_shuffle)
        return loss, pred, len_keep


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks