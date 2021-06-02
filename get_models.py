import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import join, isfile
import seqNet

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def get_model(opt,encoder_dim,device):
    model = nn.Module()

    if opt.seqL == 1 and opt.pooling.lower() not in ['single', 'seqnet']:
        raise Exception("For sequential matching/pooling, set seqL > 1")
    elif opt.seqL != 1 and opt.pooling.lower() in ['single']:
        raise Exception("For single frame based evaluation, set seqL = 1")

    if opt.pooling.lower() == 'smooth':
        global_pool = nn.AdaptiveAvgPool2d((1,None))
        model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 'seqnet':
        seqFt = seqNet.seqNet(encoder_dim, opt.outDims, opt.seqL, opt.w)
        model.add_module('pool', nn.Sequential(*[seqFt, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 's1+seqmatch':
        seqFt = seqNet.seqNet(encoder_dim, opt.outDims, 1, opt.w)
        model.add_module('pool', nn.Sequential(*[seqFt, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 'delta':
        deltaFt = seqNet.Delta(inDims=encoder_dim,seqL=opt.seqL)
        model.add_module('pool', nn.Sequential(*[deltaFt, L2Norm()]))
    elif opt.pooling.lower() == 'single':
        single = nn.AdaptiveAvgPool2d((opt.seqL,None)) # shoud have no effect
        model.add_module('pool', nn.Sequential(*[single, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 'single+seqmatch':

        model.add_module('pool', nn.Sequential(*[L2Norm(dim=-1)]))
    else:
        raise ValueError('Unknown pooling type: ' + opt.pooling)

    if not opt.resume:
        model = model.to(device)

    scheduler, optimizer, criterion = None, None, None
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    return model, optimizer, scheduler, criterion