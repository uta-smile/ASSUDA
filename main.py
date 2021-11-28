import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX
import shutil
from attack import FGSM
from util import *
from nt_xent import NT_Xent
import random

def main():
    opt = TrainOptions()
    args = opt.initialize()
    _t = {'iter time' : Timer()}
    
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)   
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    else:
        pass
        
    opt.print_options(args)
    
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)
    model, optimizer = CreateModel(args)
    model_D, optimizer_D = CreateDiscriminator(args)
    
    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])
        
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))
        
    cudnn.enabled = True
    cudnn.benchmark = False
    model.cuda()
    model_D.cuda()
    model.train()
    model_D.train()
    
    contrastive_criterion = NT_Xent(batch_size=2)
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real', \
            'loss_contrastive', 'loss_seg_src_adv', 'loss_seg_trg_adv']
    _t['iter time'].tic()

    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)
        model_D.adjust_learning_rate(args, optimizer_D, i)
        # Generate adversarial examples 
        # source image
        src_img, src_lbl, _, name_src = sourceloader_iter.next()
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
        name_src = name_src[0].split('/')[-1]
        name_src = name_src.split('.')[0]

        # target image
        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, name_trg = targetloader_iter.next()
            trg_lbl = Variable(trg_lbl.long()).cuda()
        else:
            trg_img, _, name_trg = targetloader_iter.next()
        trg_img = Variable(trg_img).cuda()
        name_trg = name_trg[0].split('/')[-1]
        name_trg = name_trg.split('.')[0]
        
        fgsm = FGSM(alpha=args.alpha)
        src_img_adv = fgsm.untargeted(model, src_img, src_lbl)
        trg_img_adv = fgsm.untargeted(model, trg_img, trg_lbl)
        
        # save image
        if (i+1) % args.save_purt == 0:
            save_image(src_img.detach()[0], args.snapshot_dir, name_src, i)
            save_image(src_img_adv.detach()[0], args.snapshot_dir, name_src, i, is_adv=True)
            save_image(trg_img.detach()[0], args.snapshot_dir, name_trg, i)
            save_image(trg_img_adv.detach()[0], args.snapshot_dir, name_trg, i, is_adv=True)
        
        #####################################################################

        ### Train ###
        #####################################################################
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        for param in model_D.parameters():
            param.requires_grad = False
        
        # source image
        src_seg_score = model(src_img)
        src_seg_score_adv = model(src_img_adv)
        loss_seg_src = CrossEntropy2d(src_seg_score, src_lbl)
        loss_seg_src_adv = CrossEntropy2d(src_seg_score_adv, src_lbl)

        # target image
        trg_seg_score = model(trg_img)
        trg_seg_score_adv = model(trg_img_adv)

        loss_seg_trg = 0
        loss_seg_trg_adv = 0
        
        if args.data_label_folder_target is not None:
            loss_seg_trg += CrossEntropy2d(trg_seg_score, trg_lbl)
            loss_seg_trg_adv += CrossEntropy2d(trg_seg_score_adv, trg_lbl)

        # Contrastive loss
        loss_contrastive = contrastive_criterion(src_seg_score, trg_seg_score, 
                                                 src_seg_score_adv, trg_seg_score_adv)
        
        loss_D_trg_fake = model_D(F.softmax(trg_seg_score, dim=1), 0) + model_D(F.softmax(trg_seg_score_adv, dim=1), 0)

        loss_total =    args.lambda_adv_target * (loss_D_trg_fake) + \
                        args.lambda_contrastive * (loss_contrastive) + \
                        loss_seg_src + loss_seg_src_adv + \
                        loss_seg_trg + loss_seg_trg_adv
                        
        loss_total.backward()

        #####################
        ### Discriminator ###
        for param in model_D.parameters():
            param.requires_grad = True
        
        src_seg_score, trg_seg_score = src_seg_score.detach(), trg_seg_score.detach()
        src_seg_score_adv, trg_seg_score_adv = src_seg_score_adv.detach(), trg_seg_score_adv.detach()
        
        loss_D_src_real = model_D(F.softmax(src_seg_score, dim=1), 0) + model_D(F.softmax(src_seg_score_adv, dim=1), 0)
        loss_D_src_real.backward()
        
        loss_D_trg_real = model_D(F.softmax(trg_seg_score, dim=1), 1) + model_D(F.softmax(trg_seg_score_adv, dim=1), 1)
        loss_D_trg_real.backward()

        optimizer.step()
        optimizer_D.step()
        
        ### Print and save ###
        for m in loss:
            train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source) +str(i+1)+'.pth' ))
            torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source) +str(i+1)+'_D.pth' ))
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][lr %.6f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr'], _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()


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

if __name__ == '__main__':
    
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    os.system('rm tmp')    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))  
    
    main()