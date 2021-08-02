from __future__ import print_function
import argparse
import random, shutil, json
from os.path import join, exists
from os import makedirs

import torch
from datetime import datetime
from tqdm import tqdm

from tensorboardX import SummaryWriter
import numpy as np
import sys

from get_datasets import get_dataset, get_splits, prefix_data
from get_models import get_model
from train import train
from test import test

parser = argparse.ArgumentParser(description='seqnet')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])

# train settings
parser.add_argument('--batchSize', type=int, default=16, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=0, help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=50, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--expName', default='0', help='Unique string for an experiment')

# path settings
parser.add_argument('--runsPath', type=str, default=join(prefix_data,'runs'), help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default=join(prefix_data,'cache'), help='Path to save cache to.')
parser.add_argument('--resultsPath', type=str, default=None, help='Path to save evaluation results to when mode=test')

# test settings
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping. 0 is off.')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', choices=['test', 'train', 'val'])
parser.add_argument('--numSamples2Project', type=int, default=-1, help='TSNE uses these many samples ([:n]) to project data to 2D; set to -1 to disable')
parser.add_argument('--extractOnly',  action='store_true', help='extract descriptors')
parser.add_argument('--predictionsFile', type=str, default=None, help='path to prior predictions data')
parser.add_argument('--seqL_filterData', type=int, help='during testing, db and qry inds will be removed that violate sequence boundaries for this given sequence length')

# dataset, model etc.
parser.add_argument('--dataset', type=str, default='nordland-sw', help='Dataset to use', choices=['nordland-sw', 'nordland-sf', 'oxford-v1.0'])
parser.add_argument('--pooling', type=str, default='seqnet', help='type of pooling to use', choices=[ 'seqnet', 'smooth', 'delta', 'single','single+seqmatch', 's1+seqmatch'])
parser.add_argument('--seqL', type=int, default=5, help='Sequence Length')
parser.add_argument('--w', type=int, default=3, help='filter size for seqNet')
parser.add_argument('--outDims', type=int, default=None, help='Output descriptor dimensions')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--descType', type=str, default="netvlad-pytorch", help='underlying descriptor type')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

if __name__ == "__main__":

    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'optim', 'margin', 'seed', 'patience', 'outDims', 'w']
    if opt.pooling.lower() != 's1+seqmatch':
        restore_var = restore_var + ['pooling']
    if opt.resume:
        if opt.pooling.lower() in ['single', 'smooth', 'delta', 'single+seqmatch']:
            raise Exception("Use pooling 'seqnet' with resume")
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    dataset, encoder_dim = get_dataset(opt)
    whole_train_set, whole_training_data_loader, train_set, whole_test_set = get_splits(opt, dataset)

    print('===> Building model')
    model, optimizer, scheduler, criterion = get_model(opt, encoder_dim, device)

    unique_string = datetime.now().strftime('%b%d_%H-%M-%S')+'_l'+str(opt.seqL)+'_'+ opt.expName
    writer = None

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recallsOrDesc, dbEmb, qEmb, rAtL, preds = test(opt, model, encoder_dim, device, whole_test_set, writer, epoch, extract_noEval=opt.extractOnly)
        if opt.resultsPath is not None:
            if not exists(opt.resultsPath):
                makedirs(opt.resultsPath)
            if opt.extractOnly:
                gt = whole_test_set.get_positives()
                numDb = whole_test_set.dbStruct.numDb
                np.savez(join(opt.resultsPath,unique_string),dbDesc=recallsOrDesc[:numDb],qDesc=recallsOrDesc[numDb:],gt=gt)
            else:
                np.savez(join(opt.resultsPath,unique_string),args=opt.__dict__,recalls=recallsOrDesc, dbEmb=dbEmb,qEmb=qEmb,rAtL=rAtL,preds=preds)

    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath,unique_string))
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache_{}.hdf5'.format(unique_string))
        if not exists(opt.cachePath):
            makedirs(opt.cachePath)

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            train(opt, model, encoder_dim, device, dataset, criterion, optimizer, train_set, whole_train_set, whole_training_data_loader, epoch, writer)
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(opt, model, encoder_dim, device, whole_test_set, writer, epoch)[0]
                is_best = recalls[5] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : False,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
