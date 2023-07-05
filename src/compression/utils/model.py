import torch, numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, n_downconv = 3, in_chn = 3):
        super().__init__()

        # a tunable number of DownConv blocks in the architecture
        self.n_downconv = n_downconv

        layer_list = [ # The two mandatory initial layers
            nn.Conv2d(in_channels=in_chn, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()

        ]
        # 'n_downconv' number of DownConv layers (In the CVPR paper, it was 3)
        for i in range(self.n_downconv):
            layer_list.extend([
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), 
                    nn.ReLU(),
                    
                ])

        layer_list.append( # The one mandatory end layer
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
        )

        # register the Sequential module
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.encoder(x), 0, 1)

class decoder(nn.Module):
    def __init__(self, n_upconv = 3, out_chn = 3):
        super().__init__()

        # a tunable number of DownConv blocks in the architecture
        self.n_upconv = n_upconv

        layer_list = [ # The one mandatory initial layers
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        ]
        # 'n_upconv' number of UpConv layers (In the CVPR paper, it was 3)
        for i in range(self.n_upconv):
            layer_list.extend([
                    nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(),
                    nn.PixelShuffle(2),
                ])
        # The mandatory final layer
        layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2)
            ])

        # register the Sequential module
        self.decoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.decoder(x), 0, 1)

class autoencoder(nn.Module):
    def __init__(self, n_updownconv = 3, in_chn = 3, out_chn = 3):
        super().__init__()
        self.n_updownconv = n_updownconv
        self.in_chn = in_chn
        self.out_chn = out_chn

        # there must be same number of 'n_downconv' and 'n_upconv'
        self.encoder = encoder(n_downconv = self.n_updownconv,in_chn=self.in_chn)
        self.decoder = decoder(n_upconv = self.n_updownconv, out_chn=self.out_chn)

    def forward(self, x):
        self.shape_input = list(x.shape) # for calculating BPP
        x = self.encoder(x)
        # print(x.shape)
        # print(torch.unique(x))
        self.shape_latent = list(x.shape) # for calculating BPP
        x = self.decoder(x)
        return x
    
