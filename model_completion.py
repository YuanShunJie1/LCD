"""
MixMatch Pytorch
"""
from __future__ import print_function
import argparse
import ast
import os
import shutil
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import models.bottom_model_plus as models
from my_utils import AverageMeter, accuracy, mkdir_p, precision_recall
import datasets.get_dataset as get_dataset
import dill
import copy
from vfl_framework import VflFramework

parser = argparse.ArgumentParser(description='Model Completion')
# dataset paras
parser.add_argument('--dataset-name', default="cifar10", type=str)
parser.add_argument('--dataset-path', default='/home/shunjie/codes/defend_label_inference/cs/Datasets/cifar10/', type=str)

# attacker's knowledge paras
parser.add_argument('--n-labeled', type=int, default=400, help='Number of labeled data')  # cifar-40, liver-10, TinyImageNet-600
# inference head paras
parser.add_argument('--num-layer', type=int, default=1, help='number of layers of the inference head')
parser.add_argument('--use-bn', type=ast.literal_eval, default=True, help='Inference head use batchnorm or not')
parser.add_argument('--activation_func_type', type=str, default='ReLU')
# vfl paras
parser.add_argument('--party-num', help='party-num', type=int, default=2)
parser.add_argument('--half', type=int, default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32

parser.add_argument('--resume-dir',  default='/home/shunjie/codes/defend_label_inference/cs/baselines/saved_models/',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint', )
parser.add_argument('--resume-name', default='cifar10_saved_framework_vfl_framework_lr=0.01_mal_mir=0.1_half=14_batch_size=128.pth',
                    type=str, metavar='NAME',
                    help='file name of the latest checkpoint', )
parser.add_argument('--out', default=None,  # 'result'
                    help='Directory to output the best checkpoint')
# evaluation paras
parser.add_argument('--k', help='top k accuracy', type=int, default=5)
# training paras
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='train batchsize')  # CIFAR10/ImageNet-32
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')  # cifar-5, BC_IDC-1, liver-5
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')  # CIFAR10 2e-3

parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')

parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--val-iteration', type=int, default=1024, help='Number of labeled data')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=50, type=float)
parser.add_argument('--T', default=0.8, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
# print paras
parser.add_argument('--print-to-txt', default=1, type=int, choices=[0, 1], help='save all outputs to txt or not')

args = parser.parse_args()
args.resume = args.resume_dir + f'{args.dataset_name}_saved_models/' + args.resume_name

state = {k: v for k, v in args._get_kwargs()}

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(args.gpu_id)
device = torch.device(f'cuda:{args.gpu_id}')

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

inference_head_setting_str = f'_layer={args.num_layer}_func={args.activation_func_type}_bn={args.use_bn}_nlabeled={args.n_labeled}'
txt_name = 'model_completion_' + args.resume_name + inference_head_setting_str + '.txt'
results_log = open(args.resume_dir +  f'{args.dataset_name}_saved_models/' + txt_name, 'w+')


if args.dataset_name in ['mnist']:
    args.lr = 1e-2
elif args.dataset_name in ['cifar10']:
    args.lr = 1e-2
elif args.dataset_name in ['cifar100']:
    args.lr = 1e-2
elif args.dataset_name in ['imagenet12']:
    args.lr = 1e-2
elif args.dataset_name in ['yeast']:
    args.lr = 1e-2
elif args.dataset_name in ['letter']:
    args.lr = 1e-2


def main():
    if args.batch_size > args.n_labeled:
        raise Exception("args.batch_size must be smaller than args.n_labeled")
    
    # datasets settings
    print('==> Preparing {}'.format(args.dataset_name))
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    size_bottom_out = dataset_setup.size_bottom_out

    num_classes = dataset_setup.num_classes
    clip_function = dataset_setup.clip_one_party_data
    zip_ = get_dataset.get_datasets_for_ssl(dataset_name=args.dataset_name, file_path=args.dataset_path, n_labeled=args.n_labeled, party_num=args.party_num)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = zip_
    
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    dataset_bs = args.batch_size * 10
    
    test_loader = data.DataLoader(test_set, batch_size=dataset_bs, shuffle=False, num_workers=0)
    train_complete_trainloader = data.DataLoader(train_complete_dataset, batch_size=dataset_bs, shuffle=True, num_workers=0, drop_last=True)

    print("==> creating bottom model plus")
    def create_model(ema=False, size_bottom_out=10, num_classes=10):
        model = models.BottomModelPlus(args.dataset_name, size_bottom_out, num_classes, num_layer=args.num_layer, activation_func_type=args.activation_func_type, use_bn=args.use_bn)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(ema=False, size_bottom_out=size_bottom_out, num_classes=num_classes)
    ema_model = create_model(ema=True, size_bottom_out=size_bottom_out, num_classes=num_classes)

    cudnn.benchmark = True

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    # Resume
    print('==> Resuming from checkpoint..')
    print("args.resume:", args.resume)
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(args.resume, pickle_module=dill)
    print("checkpoint:", checkpoint.malicious_bottom_model_a)
    model.bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)
    ema_model.bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)

    model = model.cuda()
    ema_model = ema_model.cuda()
    # test_accs = []
    print("---Label inference on complete training dataset:")

    for epoch in range(args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if args.dataset_name in ['yeast', 'letter']:
            train_table(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda, clip_function, num_classes)
        else:
            train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda, clip_function, num_classes)
        
        print("---Label inference on complete training dataset:")
        _, top_1_acc_train, top_k_acc_train = validate(train_complete_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats',  num_classes=num_classes, clip_function=clip_function)
        
        print("\n---Label inference on testing dataset:")
        _, top_1_acc_test, top_k_acc_test   = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats', num_classes=num_classes, clip_function=clip_function)

        results_log.write('LIA Train Top-1:%.2f, Top-5:%.2f   LIA Test: Top-1:%.2f, Top-5:%.2f\n'%(top_1_acc_train, top_k_acc_train, top_1_acc_test, top_k_acc_test))
        results_log.flush()   


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, clip_function, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    model.train()

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x= next(labeled_train_iter)
        try:
            inputs_u, _ = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        inputs_x = clip_function(inputs_x, args.half)
        inputs_u = clip_function(inputs_u, args.half)
        inputs_x = inputs_x.type(torch.float)
        inputs_u = inputs_u.type(torch.float)

        batch_size = inputs_x.size(0)

        targets_x = targets_x.view(-1, 1).type(torch.long)
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x, 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()

        with torch.no_grad():
            targets_x.view(-1, 1).type(torch.long)  # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            p = torch.softmax(outputs_u, dim=1)
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch + batch_idx / args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print("batch_idx:", batch_idx, " loss:", losses.avg)
    return losses.avg, losses_x.avg, losses_u.avg


def train_table(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, clip_function, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # losses_x = AverageMeter()
    # losses_u = AverageMeter()
    # ws = AverageMeter()
    end = time.time()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    model.train()

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x= next(labeled_train_iter)
        try:
            inputs_u, _ = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, _ = next(unlabeled_train_iter)

        inputs_x = clip_function(inputs_x, args.half)
        # inputs_u = clip_function(inputs_u, args.half)
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
        loss = F.cross_entropy(model(inputs_x), targets_x)

        losses.update(loss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print("batch_idx:", batch_idx, " loss:", losses.avg)
    return losses.avg

# /home/shunjie/VFL_EXP/class_shrink/Code/saved_experiment_results/saved_models/

def validate(valloader, model, criterion, epoch, use_cuda, mode, num_classes, clip_function):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = clip_function(inputs, args.half)

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            inputs = inputs.type(torch.float)
            outputs = model(inputs)
            targets = targets.type(torch.long)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, preck = accuracy(outputs, targets, topk=(1, args.k))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            topk.update(preck.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
                
                
    print("Dataset Overall Statistics:")
    if num_classes == 2:
        print("  precision", precision.avg, end='')
        print("  recall", recall.avg, end='')
        if (precision.avg + recall.avg) != 0:
            print("  F1", 2 * (precision.avg * recall.avg) / (precision.avg + recall.avg), end='')
        else:
            print(f"F1:0")
    print("top 1 accuracy:{}, top {} accuracy:{}".format(top1.avg, args.k, topk.avg))
    return losses.avg, top1.avg, topk.avg


def save_checkpoint(state, is_best, checkpoint=args.out, filename=f'{args.dataset_name}_mc_checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'{args.dataset_name}_mc_best.pth'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()

