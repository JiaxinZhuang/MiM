import monai
import torch.nn as nn


class ViT(nn.Module):
    '''ViT model.'''
    def __init__(self, args=None):
        '''Initialization.'''
        super().__init__()
        self.args = args
        self.model = monai.networks.nets.vit.ViT(in_channels=args.in_channels, num_classes=args.out_channels, patch_size=16,
                                                 img_size=(args.roi_x, args.roi_y, args.roi_z), pos_embed_type='sincos', classification=True,
                                                 post_activation="")

    def forward(self, x):
        '''Forward pass.'''
        out, _ = self.model(x)
        return out