# pppad_ver0.py
# before including ARCH = 'conv'

import torch
import torch.nn as nn
#import models.pppad_prm as prm

REF_PPPAD = (2, 3) # default values for reference range in pp-pad (hp, wp). wp shold be odd value. Imported from pppad_prm.

# Padding Calculation Class for PP-Pad
class calc_padding(nn.Module):
#    def __init__(self, padding, channels, ref = prm.REF_PPPAD):
    def __init__(self, padding, in_channels, ref = REF_PPPAD):
        # ref = (hp, wp) for top padding
        self.ref = ref
        super(calc_padding, self).__init__()
        self.padding = padding

        input_channels = ref[0] #2
        hidden_channels = 8 
        output_channels = 1

        self.conv_mlp_top = conv_mlp_pppad(input_channels, hidden_channels, output_channels, ref[1])
        self.conv_mlp_bottom = conv_mlp_pppad(input_channels, hidden_channels, output_channels, ref[1])
        self.conv_mlp_left = conv_mlp_pppad(input_channels, hidden_channels, output_channels, ref[1])
        self.conv_mlp_right = conv_mlp_pppad(input_channels, hidden_channels, output_channels, ref[1])

    def forward(self, x):
        ref = self.ref
        p = self.padding
        if p==0:
            return x
        else:
            imsize = x.shape
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            xp = torch.zeros((imsize[0],imsize[1],imsize[2]+2*p,imsize[3]+2*p), device=torch.device(device))
            #xp = xp.to(device)
            xp[:,:,p:imsize[2]+p,p:imsize[3]+p] = x

            for i in range(p): # loop for padding size [bach, chanel, hight, wigth]
                xp_top = xp[:, :, p-i:p+ref[0]-i, 0:imsize[3]+2*p].clone()
                xp_top = torch.permute(xp_top, (0, 2, 3, 1))
                xp[:, :, p-1-i:p-i, 1:imsize[3]+2*p-1] = torch.permute(self.conv_mlp_top(xp_top), (0, 3, 1, 2))

                xp_bottom = xp[:, :, p+imsize[2]-1-ref[0]+i:p+imsize[2]-1+i, 0:imsize[3]+2*p].clone()
                #xp_bottom = xp[:, :, p+imsize[2]-ref[0]+i:p+imsize[2]+i, 0:imsize[3]+2*p].clone()
                xp_bottom = torch.permute(xp_bottom, (0, 2, 3, 1))
                xp[:, :, p+imsize[2]+i:p+imsize[2]+i+1, 1:imsize[3]+2*p-1] = torch.permute(self.conv_mlp_bottom(xp_bottom), (0, 3, 1, 2))

                xp_left = xp[:, :, 0:imsize[2]+2*p, p-i:p+ref[0]-i].clone()
                xp_left = torch.permute(xp_left, (0, 3, 2, 1))
                xp[:, :, 1:imsize[2]+2*p-1, p-1-i:p-i] = torch.permute(self.conv_mlp_left(xp_left), (0, 3, 2, 1))

                xp_right = xp[:, :, 0:imsize[2]+2*p, imsize[3]+2*p-1-(ref[0]+p)+i:imsize[3]+2*p-1-p+i].clone()
                #xp_right = xp[:, :, 0:imsize[2]+2*p, p+imsize[3]-ref[0]+i:p+imsize[3]+i].clone()
                xp_right = torch.permute(xp_right, (0, 3, 2, 1))
                xp[:, :, 1:imsize[2]+2*p-1, p+imsize[3]+i:p+imsize[3]+i+1] = torch.permute(self.conv_mlp_right(xp_right), (0, 3, 2, 1))
            return xp

# Implementation of MLP with Conv for pp-pad
class conv_mlp_pppad(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, wp=3):
        super(conv_mlp_pppad, self).__init__()
        pad_wp = int((wp - 1) / 2) - 1 # wp should be odd value.
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=(wp,1), stride=1, padding=(pad_wp, 0), dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        outputs = self.classifier(x)
        return outputs
