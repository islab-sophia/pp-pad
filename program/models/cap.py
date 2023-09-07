import torch
import torch.nn as nn

CAP_REF_RANGE = 20 # default values for reference range in cap hp.

# Padding Calculation Class for CAP (context-aware padding)
# https://arxiv.org/abs/2109.07854
class CAP(nn.Module):
    def __init__(self):
        super(CAP, self).__init__()
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64, **dict(momentum=0.9997))
        self.conv1_2 = nn.Conv2d(
            64, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn_2 = nn.BatchNorm2d(64, **dict(momentum=0.9997))

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128, **dict(momentum=0.9997))
        self.conv2_2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_bn_2 = nn.BatchNorm2d(128, **dict(momentum=0.9997))
        #pool_2

        self.bottleneck = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(128, **dict(momentum=0.9997))
        self.bottleneck_2 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn_2 = nn.BatchNorm2d(128, **dict(momentum=0.9997))

        self.deconv1 = nn.Conv2d(
            256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(128, **dict(momentum=0.9997))
        self.deconv1_2 = nn.Conv2d(
            128, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv1_bn_2 = nn.BatchNorm2d(128, **dict(momentum=0.9997))

        self.deconv2 = nn.Conv2d(
            192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(64, **dict(momentum=0.9997))
        self.deconv2_2 = nn.Conv2d(
            64, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn_2 = nn.BatchNorm2d(64, **dict(momentum=0.9997))

        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        input = x
        input_size = tuple(x.size()[2:4])
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.conv1_bn_2(x)
        conv1 = self.relu(x)
        x = self.pool(conv1)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.conv2_bn_2(x)
        conv2 = self.relu(x)
        x = self.pool(conv2)

        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_bn_2(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)
        x = self.deconv1_2(x)
        x = self.deconv1_bn_2(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)
        x = self.deconv2_2(x)
        x = self.deconv2_bn_2(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = nn.functional.tanh(x)

        return x

class calc_padding(nn.Module):
    def __init__(self, padding):
        super(calc_padding, self).__init__()
        self.padding = padding


class calc_padding_CAP(nn.Module):
    def __init__(self, padding, padding_mode = 'cap_pretrain', ref = CAP_REF_RANGE):
        super(calc_padding_CAP, self).__init__()
        self.padding = padding
        self.padding_mode = padding_mode
        self.ref = ref

        self.CAP_up = CAP()
        self.CAP_down = CAP()
        self.CAP_left = CAP()
        self.CAP_right = CAP()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        ref = self.ref
        p = self.padding
        if p==0:
            return x
        else:
            imsize = x.shape
            if self.padding_mode == 'cap_pretrain':
                xp = torch.zeros((imsize[0],imsize[1],imsize[2]+2*p,imsize[3]+2*p), device=torch.device('cuda'))
                xj = torch.zeros((imsize[0],imsize[1],imsize[2],imsize[3]), device=torch.device('cuda'))  # prediction of padding values
                xt = torch.zeros((imsize[0],imsize[1],imsize[2],imsize[3]), device=torch.device('cuda'))  # ground truth values
                xz = x.clone()
                xz_2 = x.clone()
            elif self.padding_mode == 'cap_train':
                xp = torch.zeros((imsize[0],imsize[1],imsize[2]+2*p,imsize[3]+2*p), device=torch.device('cuda'))
                xj = torch.zeros((imsize[0],imsize[1],imsize[2]+2*p,imsize[3]+2*p), device=torch.device('cuda')) # prediction of padding values
            xm = torch.zeros((imsize[0],imsize[1],imsize[2]-2*p,imsize[3]-2*p), device=torch.device('cuda'))

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            xp = xp.to(device)
            xp[:,:,p:imsize[2]+p,p:imsize[3]+p] = x
            xj = xj.to(device)
            if self.padding_mode == 'cap_pretrain':
                xj[:,:,p:imsize[2]-p,p:imsize[3]-p] = xz[:,:,p:imsize[2]-p,p:imsize[3]-p]
                xt = xt.to(device)
                xt[:,:,:,:] = xz_2
            elif self.padding_mode == 'cap_train':
                xj[:,:,p:imsize[2]+p,p:imsize[3]+p] = x

            # [bach?, chanel, hight, wigth]
            xj_top = xj[:,:,0:ref,:].clone()
            xj[:,:,0:ref,:] = self.CAP_up(xj_top)
            xj_down = xj[:,:,imsize[2]-ref:imsize[2],:].clone()
            xj[:,:,imsize[2]-ref:imsize[2],:] = self.CAP_down(xj_down)
            xj_left = xj[:,:,:,0:ref].clone()
            xj[:,:,:,0:ref] = self.CAP_left(xj_left)
            xj_right = xj[:,:,:,imsize[3]-ref:imsize[3]].clone()
            xj[:,:,:,imsize[3]-ref:imsize[3]] = self.CAP_right(xj_right)

            xj[:,:,p:imsize[2]-p,p:imsize[3]-p] = xm * xj[:,:,p:imsize[2]-p,p:imsize[3]-p]
            xj = torch.tensor(xj, requires_grad=True)

            if self.padding_mode == 'cap_pretrain':
                xt[:,:,p:imsize[2]-p,p:imsize[3]-p] = xm * xt[:,:,p:imsize[2]-p,p:imsize[3]-p]
                cap_loss = self.criterion(xj, xt)
                return [xp, cap_loss]
            else:
                return xp + xj

