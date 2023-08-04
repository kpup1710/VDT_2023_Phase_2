import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torchvision

class ssm(nn.Module):
    def __init__(self, Nh, Nw, bs, ptsz = 32, pout = 512):
        super(ssm, self).__init__()

        self.Nh = Nh
        self.Nw = Nw
        self.bs = bs
        self.ptsz = ptsz
        self.pout = pout
        self.base_encoder = torchvision.models.resnet18(pretrained=False)
        self.base_encoder.fc = nn.Identity()

        self.proj1 = nn.Sequential(*[nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.pout),
                                    nn.BatchNorm1d(self.pout)])

        self.proj2 = nn.Sequential(*[nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.pout),
                                    nn.BatchNorm1d(self.pout)])
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cntH = self.Nh//(self.ptsz//2) - 1
        self.cntW = self.Nw//(self.ptsz//2) - 1
    
    def forward(self, x):
        x = x.view((-1,3,self.ptsz,self.ptsz))
        x = self.base_encoder(x)
        # print(x.shape)
        x_out = x.view((self.bs, -1, self.cntH, self.cntW))
        print(x_out.shape)
        # print(x.shape)
        #x = self.proj1(x)
        # print(x.shape)
        x = self.gap(x_out).squeeze()
        print(x.shape)
        x1 = self.proj1(x)
        # print(x.shape)
        x2 = self.proj2(x)
        
        return x1, x2, x_out
    
if __name__ == '__main__':
    model = ssm(Nh=224,Nw=224,bs=16)
    x = torch.randn((16,169,3,32, 32))
    a, b, c = model(x)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    # x = np.random.randn(224,224,3)
    # y = model.__getpatches__(x)
    # print(y.shape)