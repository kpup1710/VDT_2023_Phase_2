import torch
import torch.nn as nn
import numpy as np
from Transformers import *

class Con_Transfromer(nn.Module):
    def __init__(self, num_tokens=169, dim_in=512,  dim=512, heads = 8, dim_head = 64, dropout = 0.) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.dim_in = dim_in
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.conv = nn.Sequential(
                     nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim_in, kernel_size=2),
                     nn.Conv2d(in_channels=self.dim_in, out_channels=self.dim, kernel_size=2),
                     nn.MaxPool2d(stride=1, kernel_size=2),
                     nn.AdaptiveAvgPool2d(1)
        )

        self.pos_embedding1 = nn.Parameter(torch.randn(1, self.num_tokens, self.dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, self.num_tokens + 1, self.dim))
        self.cls_token_q = nn.Parameter(torch.randn(1, 1, self.dim))
        self.cls_token_r = nn.Parameter(torch.randn(1, 1, self.dim))

        self.sa = self_Attention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, dropout=self.dropout)
        self.ca_r = cross_Attention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, dropout=self.dropout)
        self.ca_q = cross_Attention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, dropout=self.dropout)

    def forward(self, pFr, pFq):
        f_c_r = self.conv(pFr)
        f_c_q = self.conv(pFq)

        Fr = rearrange(pFr, 'b c h w  -> b (h w) c')
        Fq = rearrange(pFq, 'b c h w  -> b (h w) c')

        # print(Fr.shape)
        # print(self.cls_token_r.shape)
        cls_token_r = repeat(self.cls_token_r, '1 1 d -> b 1 d', b = Fr.shape[0])
        cls_token_q = repeat(self.cls_token_q, '1 1 d -> b 1 d', b = Fq.shape[0])
        pos_Fr = Fr + self.pos_embedding1
        pos_Fq = Fq + self.pos_embedding1

        cls_Fr = torch.cat([cls_token_r, Fr], dim=1)
        cls_Fq = torch.cat([cls_token_q, Fq], dim=1)

        pos_cls_Fr = cls_Fr + self.pos_embedding2 
        pos_cls_Fq = cls_Fq + self.pos_embedding2 

        f_cls_r = self.sa(pos_cls_Fr)[:,0]
        f_cls_q = self.sa(pos_cls_Fq)[:,0]

        f_p_r = self.ca_r(f_cls_q, pos_Fr)
        f_p_q = self.ca_q(f_cls_r, pos_Fq)

        return f_p_r, f_p_q, f_c_r.squeeze(-1).squeeze(-1), f_c_q.squeeze(-1).squeeze(-1)
    
if __name__ == "__main__":
    x = torch.randn((2,512,13,13))
    y = torch.randn((2,512,13,13))

    ct = Con_Transfromer()

    a, b, c, d, = ct(x, y)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    