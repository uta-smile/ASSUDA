import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
from util import *
from PIL import Image


def CrossEntropy2d(predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss


class FGSM():
    def __init__(self, alpha=1, eps=1): 
        self.eps=eps
        self.alpha=alpha
        self.predictor=None
        self.default_its=min(int(self.eps+4),int(1.25*self.eps))
    
    def untargeted(self, model, img, label):
        adv_bx = img.detach()
        adv_bx.requires_grad_()

        with torch.enable_grad():
            logits = model(adv_bx)
            loss = CrossEntropy2d(logits, label)

        grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]
        adv_bx = adv_bx.detach() + self.alpha * torch.sign(grad.detach())
        adv_bx = adv_bx.clamp_(-255, 255)
        return adv_bx
