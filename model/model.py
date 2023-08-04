import torch
import torch.nn as nn
from .contrast_Decoder import Con_Transfromer
from .self_sl import ssm

class Model(nn.Module):
    def __init__(self, Nh=224, Nw=224, bs=32, ptsz = 32, pout = 512, num_tokens=169, dim_in=512,  dim=512, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.Nh = Nh
        self.Nw = Nw
        self.bs = bs
        self.ptsz = ptsz
        self.pout = pout
        self.num_tokens = num_tokens
        self.dim_in = dim_in
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.backbone = ssm(self.Nh, self.Nw, self.bs, self.ptsz, self.pout)
        self.con_trans = Con_Transfromer(self.num_tokens, self.dim_in, self.dim, self.heads, self.dim_head, self.dropout)

    def forward(self, query, reference):
        _,_,Fq = self.backbone(query)
        _,_,Fr = self.backbone(reference)

        f_p_r, f_p_q, f_c_r, f_c_q = self.con_trans(Fr, Fq)

        return f_p_r, f_p_q, f_c_r, f_c_q
    
if __name__ == "__main__":
    model = Model()
    x = torch.randn((32,169,3,32, 32))
    y = torch.randn((32,169,3,32, 32))

    a, b, c, d = model(x, y)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)