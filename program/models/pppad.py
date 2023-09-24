# pppad_ver1.py
# including ARCH = 'conv' and 'pppad'

import torch
import torch.nn as nn
import models.pppad_prm as prm

# hyper-parameters for pp-pad specified in pppad_prm.py
# REF_PPPAD = (2, 3) # default values for reference range in pp-pad (hp, wp). wp shold be odd value.
# ARCH = 'pppad' # 'pppad' or 'conv'
# HIDDEN_CHANNELS = 8

# Padding Calculation Class for PP-Pad
class calc_padding(nn.Module):
    def __init__(self, padding, in_channels, ref = prm.REF_PPPAD, hidden_channels = prm.HIDDEN_CHANNELS):
        super(calc_padding, self).__init__()
        self.ref = ref
        if prm.ARCH == 'pppad':
            self.calc_ppading = calc_padding_pppad(padding, ref, hidden_channels)
        elif prm.ARCH == 'conv':
            self.calc_ppading = calc_padding_conv(padding, in_channels, ref, hidden_channels)
    
    def forward(self, x):
        return self.calc_ppading(x)

# Implementation of pp-pad with MLP using conv
class calc_padding_pppad(nn.Module):
    def __init__(self, padding, ref = prm.REF_PPPAD, hidden_channels = 8):
        super(calc_padding_pppad, self).__init__()
        # ref = (hp, wp) for top padding
        self.ref = ref
        self.padding = padding

        input_channels = ref[0] #2
        hidden_channels = hidden_channels #8
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

                #xp_bottom = xp[:, :, p+imsize[2]-1-ref[0]+i:p+imsize[2]-1+i, 0:imsize[3]+2*p].clone()
                xp_bottom = xp[:, :, p+imsize[2]-ref[0]+i:p+imsize[2]+i, 0:imsize[3]+2*p].clone()
                xp_bottom = torch.permute(xp_bottom, (0, 2, 3, 1))
                xp[:, :, p+imsize[2]+i:p+imsize[2]+i+1, 1:imsize[3]+2*p-1] = torch.permute(self.conv_mlp_bottom(xp_bottom), (0, 3, 1, 2))

                xp_left = xp[:, :, 0:imsize[2]+2*p, p-i:p+ref[0]-i].clone()
                xp_left = torch.permute(xp_left, (0, 3, 2, 1))
                xp[:, :, 1:imsize[2]+2*p-1, p-1-i:p-i] = torch.permute(self.conv_mlp_left(xp_left), (0, 3, 2, 1))

                #xp_right = xp[:, :, 0:imsize[2]+2*p, imsize[3]+2*p-1-(ref[0]+p)+i:imsize[3]+2*p-1-p+i].clone()
                xp_right = xp[:, :, 0:imsize[2]+2*p, p+imsize[3]-ref[0]+i:p+imsize[3]+i].clone()
                xp_right = torch.permute(xp_right, (0, 3, 2, 1))
                xp[:, :, 1:imsize[2]+2*p-1, p+imsize[3]+i:p+imsize[3]+i+1] = torch.permute(self.conv_mlp_right(xp_right), (0, 3, 2, 1))
        
            return xp

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


# Implementation of pp-pad with normal conv
class calc_padding_conv(nn.Module):
    def __init__(self, padding, in_channels, ref = prm.REF_PPPAD, hidden_channels = 8):
        super(calc_padding_conv, self).__init__()
        # ref = (hp, wp) for top padding
        self.ref = ref
        self.padding = padding

        input_channels = in_channels
        hidden_channels = hidden_channels
        output_channels = 1

        self.conv_top = conv_pppad(input_channels, hidden_channels, output_channels, ref, 'lateral')
        self.conv_bottom = conv_pppad(input_channels, hidden_channels, output_channels, ref, 'lateral')
        self.conv_left = conv_pppad(input_channels, hidden_channels, output_channels, ref, 'vertical')
        self.conv_right = conv_pppad(input_channels, hidden_channels, output_channels, ref, 'vertical')

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
                xp[:, :, p-1-i:p-i, 1:imsize[3]+2*p-1] = self.conv_top(xp_top)

                xp_bottom = xp[:, :, p+imsize[2]-ref[0]+i:p+imsize[2]+i, 0:imsize[3]+2*p].clone()
                xp[:, :, p+imsize[2]+i:p+imsize[2]+i+1, 1:imsize[3]+2*p-1] = self.conv_bottom(xp_bottom)

                xp_left = xp[:, :, 0:imsize[2]+2*p, p-i:p+ref[0]-i].clone()
                xp[:, :, 1:imsize[2]+2*p-1, p-1-i:p-i] = self.conv_left(xp_left)

                xp_right = xp[:, :, 0:imsize[2]+2*p, p+imsize[3]-ref[0]+i:p+imsize[3]+i].clone()
                xp[:, :, 1:imsize[2]+2*p-1, p+imsize[3]+i:p+imsize[3]+i+1] = self.conv_right(xp_right)
        
            return xp

# Implementation of pp-pad with Conv
class conv_pppad(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, ref, axis):
        super(conv_pppad, self).__init__()
        hp, wp = ref
        pad_wp = int((wp - 1) / 2) - 1 # wp should be odd value.
        if axis == 'vertical':
            kernel_size = (wp, hp)
            padding = (pad_wp, 0)
        else: #'lateral'
            kernel_size = (hp, wp)
            padding = (0, pad_wp)
        self.classifier = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        outputs = self.classifier(x)
        return outputs
