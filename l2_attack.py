import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class l2_attack(object):
    def __init__(self,model,iters,c,verbose=False):
        self.model=model
        self.model.nn=self.model.nn.cpu()
        self.iters=iters
        self.c=c
        self.verbose=verbose

    def attack(self,x,target,kappa=0):
        x = torch.Tensor(x)
        target = np.array(target)
        target = Variable(torch.Tensor(target.reshape(1, -1)), requires_grad=False)
        target = target[None:]
        w = Variable(torch.Tensor(np.arctanh((x.numpy() - 0.5) * 1.9999)), requires_grad=True)
        x = Variable(x, requires_grad=False)
        optimizer = optim.Adam([w], lr=1e-2)
        for i in range(self.iters):
            optimizer.zero_grad()
            newimg = (F.tanh(w) + 1) * 0.5
            loss1 = torch.norm(newimg - x, 2)
            f = self.model.extract(newimg[None, :])
            other = torch.max(f * (1 - target))
            real = torch.sum(f * target)
            loss2 = self.c * torch.max(other - real, Variable(torch.Tensor([-kappa])))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            if i%100==0 and self.verbose:
                print('Iters:  [{}/{}]\tLoss: {}'.format(i,self.iters,loss.data.numpy()[0]))
        return newimg
