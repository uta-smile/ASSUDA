import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


class FCN8s(nn.Module):
    def __init__(self, num_classes, phase, vgg16_caffe_path=None):
        self.phase = phase
        super(FCN8s, self).__init__()
        vgg = models.vgg16()
        if vgg16_caffe_path is not None:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        self.classifier = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)
        

    def forward(self, x, ssl=False, lbl=None):
        _, _, h, w = x.size()
        x = self.features(x)
        x = self.classifier(x)

        if self.phase == 'train' and not ssl:
            x = nn.functional.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        return x


    def get_parameters(self, bias=False):
        import torch.nn as nn
        modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            FCN8s,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                # weight is frozen because it is just a bilinear upsampling
                if bias:
                    assert m.bias is None
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))

    def get_1x_lr_params_NOscale(self):
        b = []
        b.append(self.features.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_10x_lr_params(self):
        b = []
        b.append(self.classifier.parameters())
        
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def adjust_learning_rate(self, args, optimizer, i):
        optimizer.param_groups[0]['lr'] = args.learning_rate * (0.1**(int(i/30000)))
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = args.learning_rate * (0.1**(int(i/30000))) * 2

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
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


def VGG16_FCN8s(num_classes=21, init_weights=None, restore_from=None, phase='train'):
    model = FCN8s(num_classes=num_classes, phase=phase, vgg16_caffe_path=init_weights)

    if restore_from is not None:
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))
    return model
