import argparse
import shutil
from os.path import join, isdir, isfile
from os import makedirs

from dataset import VID
from net import DCFNet
import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
import pdb

parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', '-s', default='./workUn', type=str, help='directory for saving')

args = parser.parse_args()

print args
best_loss = 1e6


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)

def output_drop(output, target):
    delta1 = (output - target)**2
    batch_sz = delta1.shape[0]
    delta = delta1.view(batch_sz, -1).sum(dim=1)
    sort_delta, index = torch.sort(delta, descending=True) 
    # unreliable samples (10% of the total) do not produce grad (we simply copy the groundtruth label)
    for i in range(int(round(0.1*batch_sz))):
        output[index[i],...] = target[index[i],...]
    return output
    
class TrackerConfig(object):
    crop_sz = 125
    output_sz = 121

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1

    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    yf = torch.rfft(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), signal_ndim=2)
    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window


config = TrackerConfig()

model = DCFNet(config=config)
model.cuda()
gpu_num = torch.cuda.device_count()
print('GPU NUM: {:2d}'.format(gpu_num))
if gpu_num > 1:
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

criterion = nn.MSELoss(size_average=False).cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)  # for training

# optionally resume from a checkpoint
if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

# training data
crop_base_path = join('dataset', 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
if not isdir(crop_base_path):
    print('please run gen_training_data.py --output_size {:d} --padding {:.1f}!'.format(args.input_sz, args.padding))
    exit()

save_path = join(args.save, 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
if not isdir(save_path):
    makedirs(save_path)

train_dataset = VID(root=crop_base_path, train=True, range=args.range)
val_dataset = VID(root=crop_base_path, train=False, range=args.range)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*gpu_num, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size*gpu_num, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

def adjust_learning_rate(optimizer, epoch):
    lr = np.logspace(-2, -5, num=args.epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename=join(save_path, 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (template, search1, search2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        template = template.cuda(non_blocking=True)
        search1 = search1.cuda(non_blocking=True)
        search2 = search2.cuda(non_blocking=True)
        label = config.yf.repeat(args.batch_size*gpu_num,1,1,1,1).cuda(non_blocking=True)
        initial_y = config.y.copy()
    
        # forward tracking 1
        with torch.no_grad():
            s1_response = model(template, search1, label)
        # label transform
        peak, index = torch.max(s1_response.view(args.batch_size*gpu_num, -1), 1)
        r_max, c_max = np.unravel_index(index, [config.output_sz, config.output_sz])
        fake_y = np.zeros((args.batch_size*gpu_num, 1, config.output_sz, config.output_sz))
        # label shift
        for j in range(args.batch_size*gpu_num):
            shift_y  = np.roll(initial_y, r_max[j])
            fake_y[j,...] = np.roll(shift_y, c_max[j])
        fake_yf = torch.rfft(torch.Tensor(fake_y).view(args.batch_size*gpu_num, 1, config.output_sz, config.output_sz).cuda(), signal_ndim = 2)
        fake_yf = fake_yf.cuda(non_blocking=True)

        # forward tracking 2
        with torch.no_grad():
            s2_response = model(search1, search2, fake_yf)
        peak, index = torch.max(s2_response.view(args.batch_size*gpu_num, -1), 1)
        r_max, c_max = np.unravel_index(index, [config.output_sz, config.output_sz])
        fake_y = np.zeros((args.batch_size*gpu_num, 1, config.output_sz, config.output_sz))
        for j in range(args.batch_size*gpu_num):
            shift_y = np.roll(initial_y, r_max[j])
            fake_y[j,...] = np.roll(shift_y, c_max[j])
        fake_yf = torch.rfft(torch.Tensor(fake_y).view(args.batch_size*gpu_num, 1, config.output_sz, config.output_sz).cuda(), signal_ndim = 2)
        fake_yf = fake_yf.cuda(non_blocking=True)
    
        # backward tracking
        output = model(search2, template, fake_yf)
        output = output_drop(output, target)  # the sample dropout is necessary, otherwise we find the loss tends to become unstable

        # consistency loss. target is the initial Gaussian label
        loss = criterion(output, target)/template.size(0)
        # measure accuracy and record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (template, search1, search2) in enumerate(val_loader):

            # compute output
            template = template.cuda(non_blocking=True)
            search1 = search1.cuda(non_blocking=True)
            search2 = search2.cuda(non_blocking=True)
            initial_y = config.y.copy()
            label = config.yf.repeat(args.batch_size*gpu_num,1,1,1,1).cuda(non_blocking=True)
  
            # forward tracking 1
            # switch to evaluate mode, therefore ''torch.no_grad()'' is unnecessary
            s1_response = model(template, search1, label)
            # label transform
            peak, index = torch.max(s1_response.view(args.batch_size*gpu_num, -1), 1)
            r_max, c_max = np.unravel_index(index, [config.output_sz, config.output_sz])
            fake_y = np.zeros((args.batch_size*gpu_num, 1, config.output_sz, config.output_sz))
            for j in range(args.batch_size*gpu_num):
                shift_y  = np.roll(initial_y, r_max[j])
                fake_y[j,...] = np.roll(shift_y, c_max[j])
            fake_yf = torch.rfft(torch.Tensor(fake_y).view(args.batch_size*gpu_num, 1, config.output_sz, config.output_sz).cuda(), signal_ndim = 2)
            fake_yf = fake_yf.cuda(non_blocking=True)

            # forward tracking 2
            s2_response = model(search1, search2, fake_yf)
            peak, index = torch.max(s2_response.view(args.batch_size*gpu_num, -1), 1)
            r_max, c_max = np.unravel_index(index, [config.output_sz, config.output_sz])
            fake_y = np.zeros((args.batch_size*gpu_num, 1, config.output_sz, config.output_sz))
            for j in range(args.batch_size*gpu_num):
                shift_y = np.roll(initial_y, r_max[j])
                fake_y[j,...] = np.roll(shift_y, c_max[j])
            fake_yf = torch.rfft(torch.Tensor(fake_y).view(args.batch_size*gpu_num, 1, config.output_sz, config.output_sz).cuda(), signal_ndim = 2)
            fake_yf = fake_yf.cuda(non_blocking=True)
   
            # backward tracking
            output = model(search2, template, fake_yf)
            output = output_drop(output, target)
            loss = criterion(output, target)/(args.batch_size * gpu_num)

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))

        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    loss = validate(val_loader, model, criterion)

    # remember best loss and save checkpoint
    is_best = loss < best_loss
    best_loss = min(best_loss, loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
