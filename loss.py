import torch
import torch.nn as nn
import torch.nn.functional as F

# loss for self-supervised
class  CorLoss(nn.Module):
    def __init__(self, batch_size=32, temperature = 0.5, lambda_loss = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # self.n_temperature = n_temperature
        self.lambda_loss = lambda_loss
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            mask[i, i] = 0
        return mask
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        if z1.shape[0] != self.batch_size:
            self.batch_size = z1.shape[0]
        N = 2 * self.batch_size
        z1 = F.normalize(z1, dim=0, p=2)
        #z1 = z1 - z1.mean(dim = 0)
        z2 = F.normalize(z2, dim=0, p=2)
        #z2 = z2 - z2.mean(dim = 0)
        z1mod = z1 - z1.mean(dim = 0)
        z2mod = z2 - z2.mean(dim = 0)
        crosscovmat = z1mod.T@z2mod
        loss = torch.square(self.off_diagonal(crosscovmat)).sum() + torch.square(torch.diag(crosscovmat)-1).sum()
        loss /= N

        return loss
    
class SparsityLoss(nn.Module):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
    
    def forward(self, r, q):
        loss = torch.mean(- torch.sum(r*torch.log(r),dim=1) -  torch.sum(q*torch.log(q),dim=1))

        # print(loss.shape)
        loss /= self.batch_size
        return loss
class SSLLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, x1, x2):
        # q1 = self.sm(x1)
        # q2 = self.sm(x2)
        # loss = -torch.sum(q2*torch.log(q1) + q1*torch.log(q2)) / x1.shape[0]
        loss = torch.mean(torch.norm(x1- x2), dim=-1)
        return loss


class NewFocalLoss(nn.Module):
    def __init__(self, alpha1=0.6, alpha2=0.8, m=0.9, n=0.3, K=10, V=10):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.m = m
        self.n = n
        self.K = K
        self.V = V
        self.sigmoid = nn.Sigmoid()

    def forward(self, fr, fq, lb):
        d = (fr - fq).pow(2).sum(1).sqrt()
        # print(d)
        lf = torch.mean((1-lb)*self.sigmoid(self.K*(d - self.alpha1)) + lb*self.sigmoid(self.V*(self.alpha2 - d)))
        return lf



if __name__ == '__main__':
    x ,y = 1.1*torch.randn((512, 128)),torch.randn((512, 128)) + 1
    lb = torch.zeros((512))
    loss = SSLLoss()
    focalloss = NewFocalLoss()
    l = loss(x, y)
    print(l)