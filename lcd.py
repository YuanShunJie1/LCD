# 
# label compression + embedding compaction
# 
import argparse
import ast
import os
import sys
sys.path.insert(0, "./")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from datasets import get_dataset
from my_utils import utils
from models import model_sets
import my_optimizers
import possible_defenses
import numpy as np
import random
from scipy.spatial.distance import cosine

from center_loss import CenterLoss
import support as sp
from support import split_data, test_per_epoch, myprint
from vfl_framework import VflFramework
import dill

# lc: label compression defense
# mnist, cifar10, cifar100, imagenet12, yeast, letter

parser = argparse.ArgumentParser(description='vfl framework training')
# dataset paras
parser.add_argument('-d', '--dataset', default='mnist', type=str, help='name of dataset')
parser.add_argument('--path-dataset', help='path_dataset', type=str, default='/home/shunjie/codes/defend_label_inference/cs/Datasets/yeast/yeast.data')
parser.add_argument('--half',  type=int, default=16)
parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA', type=ast.literal_eval, default=True)
# attack paras
parser.add_argument('--use-mal-optim',  help='whether the attacker uses the malicious optimizer', type=ast.literal_eval, default=False)
# saving path paras
parser.add_argument('--save_dir', dest='save_dir', help='The directory used to save the trained models and csv files', default='./saved_experiment_results', type=str)

parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,  metavar='LR', help='initial learning rate')  # TinyImageNet=5e-2, Yahoo=1e-3
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--step-gamma', default=0.1, type=float, metavar='S', help='gamma for step scheduler')
parser.add_argument('--stone1', default=50, type=int, metavar='s1', help='stone1 for step scheduler')
parser.add_argument('--stone2', default=85, type=int, metavar='s2', help='stone2 for step scheduler')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of datasets loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--feat_dim',  default=128, type=int)

# paras about label compression
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warm-up epochs to run')
parser.add_argument('--num_new_classes', default=5, type=int, help='number of fake labels')
parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')
parser.add_argument('--weight_cent', default=1, type=float, help='weight_cent')
parser.add_argument('--lr_cent', default=0.5, type=float, help='lr_cent')
args = parser.parse_args()

args = sp.proccess_params(args)

torch.cuda.set_device(args.gpu_id)
plt.switch_backend('agg')

if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

setting_str, dir_save_model = sp.generate_log(args)
acc_test_log = open(os.path.join(dir_save_model, setting_str),'w')

class VflFramework(nn.Module):
    def __init__(self):
        super(VflFramework, self).__init__()
        self.inferred_correct = 0
        self.inferred_wrong = 0
        self.collect_outputs_a = False
        self.outputs_a = torch.tensor([]).cuda()
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        self.if_collect_training_dataset_labels = False
        
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.criterion_cent = CenterLoss(num_classes=args.num_new_classes, feat_dim=args.feat_dim, use_gpu=True)

        self.loss_func_bottom_model = utils.keep_predict_loss
        self.malicious_bottom_model_a = model_sets.BottomModel(dataset_name=args.dataset).get_model(half=args.half,is_adversary=True)
        self.benign_bottom_model_b = model_sets.BottomModel(dataset_name=args.dataset).get_model(half=args.half,is_adversary=False)
        self.top_fake_model = model_sets.FakeTopModel(dataset_name=args.dataset, output_times=args.num_new_classes).get_model()
        self.top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()
        
        if os.path.exists(os.path.join(dir_save_model, 'true2false.pth')):
            self.true2false = torch.load(os.path.join(dir_save_model, 'true2false.pth'))
        else:
            self.true2false = {}

        if os.path.exists(os.path.join(dir_save_model, 'false2true.pth')):
            self.false2true = torch.load(os.path.join(dir_save_model, 'false2true.pth'))
        else:
            self.false2true = {}
        
        # This setting is for adversarial experiments except sign SGD
        self.optimizer_top_model = optim.SGD(self.top_model.parameters(), lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_top_fake_model = optim.SGD(self.top_fake_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_centloss = optim.SGD(self.criterion_cent.parameters(), lr=args.lr_cent)

        if args.use_mal_optim:
            self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(self.malicious_bottom_model_a.parameters(),lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
        else:
            self.optimizer_malicious_bottom_model_a = optim.SGD(self.malicious_bottom_model_a.parameters(),lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)

        self.optimizer_benign_bottom_model_b = optim.SGD(self.benign_bottom_model_b.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)


    def warmup_and_initialize_fake_labels(self, train_loader, labels):
        # In this implementation, we adopt an approximation of the DP algorithm.
        num_new_classes = args.num_new_classes
        num_classes = len(set(labels))

        act_bot_feats = None
        targets = None

        num_iter = (len(train_loader.dataset)//(args.batch_size))+1
        for epoch in range(args.warmup_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.float().cuda()
                target = target.long().cuda()

                loss_framework, act_bot_feat_batch = self.simulate_warm_up_train_round_per_batch(data, target)

                if batch_idx % 100 ==0:
                    myprint(args, epoch,batch_idx,num_iter,loss_framework.item())
                    
                if epoch == args.warmup_epochs - 1:
                    if act_bot_feats is None:
                        act_bot_feats = act_bot_feat_batch
                        targets = target
                    else:
                        act_bot_feats = torch.cat((act_bot_feats, act_bot_feat_batch), dim=0)
                        targets = torch.cat((targets, target), dim=0)

        act_bot_feats = act_bot_feats.detach().cpu()
        targets_np = targets.cpu().numpy()
        centers = []

        for c in range(num_classes):
            indices = np.where(targets_np == c)[0]
            center = torch.mean(act_bot_feats[indices], dim=0)
            centers.append(center)

        centers = torch.stack(centers).float()
        centers = F.normalize(centers, p=2, dim=1)

        centers = centers.detach().cpu().numpy()
        dist_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                dist = cosine(centers[i], centers[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        merged = np.zeros(num_classes, dtype=bool)
        self.false2true = {i: [] for i in range(num_new_classes)}
        index = 0
        
        while args.num_new_classes > index:
            np.fill_diagonal(dist_matrix, np.inf)
            dist_matrix[merged, :] = np.inf
            dist_matrix[:, merged] = np.inf

            class1, class2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            merged[class1] = True
            merged[class2] = True

            self.false2true[index].extend([class1, class2])
            index += 1

        for new_label, original_classes in self.false2true.items():
            for true_label in original_classes:
                self.true2false[true_label] = new_label

        print('Fake and ground-truth labels info:')
        print('True labels to fake labels:')
        print(self.true2false)
        print('Fake labels to true labels:')
        print(self.false2true)
        
        torch.save(self.true2false, os.path.join(dir_save_model, 'true2false.pth'))
        torch.save(self.false2true, os.path.join(dir_save_model, 'false2true.pth'))
        return 
        
    def simulate_warm_up_train_round_per_batch(self, data, target):
        # This implementation slightly differs from the description in the paper. Here, we train the entire model during the warm-up phase, whereas in the paper, only the active participant’s model is trained. However, this discrepancy does not affect the experimental results. The only potential issue is that training the full model during warm-up may leak label information on simpler datasets. We will fix this bug in the code and update it to train only the active participant’s model, in line with the algorithm described in the paper.
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)
        
        x_a, x_b = split_data(args, data)
        
        self.malicious_bottom_model_a.train(mode=True)
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        
        self.benign_bottom_model_b.train(mode=True)
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)

        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

        self.top_model.train(mode=True)
        output_framework = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)
        
        loss_framework = model_sets.update_top_model_one_batch(optimizer=self.optimizer_top_model, model=self.top_model, output=output_framework, batch_target=target, loss_func=self.loss_func_top_model)
        
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad
        
        # --bottom models backward/update--
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_malicious_bottom_model_a,
                                                 model=self.malicious_bottom_model_a,
                                                 output=output_tensor_bottom_model_a,
                                                 batch_target=grad_output_bottom_model_a,
                                                 loss_func=self.loss_func_bottom_model)
        
        # -bottom model b: backward/update-
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                 model=self.benign_bottom_model_b,
                                                 output=output_tensor_bottom_model_b,
                                                 batch_target=grad_output_bottom_model_b,
                                                 loss_func=self.loss_func_bottom_model)
        
        return loss_framework, input_tensor_top_model_b.detach()

    
    def train_with_fake_labels(self, data, target):
        if self.if_collect_training_dataset_labels:
            self.labels_training_dataset = torch.cat((self.labels_training_dataset, target), dim=0)
        # store grad of input of top model/outputs of bottom models
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)

        # --bottom models forward--
        x_a, x_b = split_data(args, data)
        
        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        # bottom model a can collect output_a for label inference attack
        if self.collect_outputs_a:
            self.outputs_a = torch.cat((self.outputs_a, output_tensor_bottom_model_a.data))
        
        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        # -top model-
        # (we omit interactive layer for it doesn't effect our attack or possible defenses)
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data
        
        self.top_fake_model.train(mode=True)
        output_framework = self.top_fake_model(input_tensor_top_model_a, input_tensor_top_model_b)
        
        loss_framework = model_sets.update_fake_top_model_and_center_loss_one_batch(optimizer=self.optimizer_top_fake_model, 
                                                                                    model=self.top_fake_model, 
                                                                                    output=output_framework, 
                                                                                    batch_target=target, 
                                                                                    loss_func=self.loss_func_top_model, 
                                                                                    criterion_cent=self.criterion_cent, 
                                                                                    optimizer_centloss=self.optimizer_centloss, 
                                                                                    bottom_features=input_tensor_top_model_a, 
                                                                                    weight_cent=args.weight_cent)
        
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad

        model_all_layers_grads_list = [grad_output_bottom_model_a, grad_output_bottom_model_b]
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(model_all_layers_grads_list)

        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_malicious_bottom_model_a,
                                                 model=self.malicious_bottom_model_a,
                                                 output=output_tensor_bottom_model_a,
                                                 batch_target=grad_output_bottom_model_a,
                                                 loss_func=self.loss_func_bottom_model)
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                 model=self.benign_bottom_model_b,
                                                 output=output_tensor_bottom_model_b,
                                                 batch_target=grad_output_bottom_model_b,
                                                 loss_func=self.loss_func_bottom_model)

        return loss_framework          

    def train_with_true_labels(self, data, target):
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)
        
        # --bottom models forward--
        x_a, x_b = split_data(args, data)
        
        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        
        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        # -top model-
        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data
        
        self.top_model.train(mode=True)
        output_framework = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)
        
        loss_framework = model_sets.update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                                model=self.top_model,
                                                                output=output_framework,
                                                                batch_target=target,
                                                                loss_func=self.loss_func_top_model)
        
        grad_output_bottom_model_b = input_tensor_top_model_b.grad
        
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                 model=self.benign_bottom_model_b,
                                                 output=output_tensor_bottom_model_b,
                                                 batch_target=grad_output_bottom_model_b,
                                                 loss_func=self.loss_func_bottom_model)
        
        return loss_framework


def set_loaders():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset, train_labels = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    test_dataset, test_labels = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)

    return train_loader, test_loader, train_labels


def main():
    vfl_lc = VflFramework()
    vfl_lc = vfl_lc.cuda()
    cudnn.benchmark = True
    
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(vfl_lc.optimizer_top_model, milestones=[args.stone1, args.stone2], gamma=args.step_gamma)
    lr_scheduler_top_fake_model = torch.optim.lr_scheduler.MultiStepLR(vfl_lc.optimizer_top_fake_model, milestones=[args.stone1, args.stone2], gamma=args.step_gamma)
    lr_scheduler_m_a = torch.optim.lr_scheduler.MultiStepLR(vfl_lc.optimizer_malicious_bottom_model_a, milestones=[args.stone1, args.stone2], gamma=args.step_gamma)
    lr_scheduler_b_b = torch.optim.lr_scheduler.MultiStepLR(vfl_lc.optimizer_benign_bottom_model_b, milestones=[args.stone1, args.stone2], gamma=args.step_gamma)
    
    train_loader, val_loader, train_labels = set_loaders()
    vfl_lc.warmup_and_initialize_fake_labels(train_loader, train_labels)

    test_loss, correct_top1, correct_topk = test_per_epoch(args, test_loader=val_loader, framework=vfl_lc, k=5, loss_func_top_model=vfl_lc.loss_func_top_model)
    print('Epoch:%d Top-1:%.2f Top-5:%.2f\n'%(0, correct_top1, correct_topk))

    sp.random_initialize(vfl_lc.top_model, init_type='kaiming_normal')
    sp.random_initialize(vfl_lc.malicious_bottom_model_a, init_type='kaiming_normal')
    sp.random_initialize(vfl_lc.benign_bottom_model_b, init_type='kaiming_normal')

    acc_test_log.write('Deception Training Stage\n')
    acc_test_log.flush() 

    for epoch in range(args.epochs):
        if epoch == args.epochs - 1 and args.if_cluster_outputsA:
            vfl_lc.collect_outputs_a = True
            vfl_lc.if_collect_training_dataset_labels = True
        
        num_iter = (len(train_loader.dataset)//(args.batch_size))+1
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().cuda()
            targets = [vfl_lc.true2false[label.item()] for label in target]
            targets = torch.LongTensor(targets).cuda()
            loss_framework = vfl_lc.train_with_fake_labels(data, targets)
            if batch_idx % 100 ==0:
                myprint(args, epoch,batch_idx,num_iter,loss_framework.item())
        
        lr_scheduler_top_fake_model.step()
        lr_scheduler_m_a.step()
        lr_scheduler_b_b.step()

        test_loss, correct_top1, correct_topk = test_per_epoch(args, test_loader=val_loader, framework=vfl_lc, k=5, loss_func_top_model=vfl_lc.loss_func_top_model)
        print('Epoch:%d Top-1:%.2f Top-5:%.2f\n'%(epoch, correct_top1, correct_topk))
        acc_test_log.write('Epoch:%d Loss:%.2f Top-1:%.2f Top-5:%.2f\n'%(epoch, test_loss, correct_top1, correct_topk))
        acc_test_log.flush() 

    acc_test_log.write('Recovery Training Stage\n')
    acc_test_log.flush() 
    print('Training the top model, while fixing bottom models!')
    for new_epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().cuda(), target.long().cuda()
        
            loss_framework = vfl_lc.train_with_true_labels(data, target)
            myprint(args, new_epoch, batch_idx,num_iter,loss_framework.item())
            
        lr_scheduler_top_model.step()
        lr_scheduler_b_b.step()
        
        test_loss, correct_top1, correct_topk = test_per_epoch(args, test_loader=val_loader, framework=vfl_lc, k=5, loss_func_top_model=vfl_lc.loss_func_top_model)
        print('Epoch:%d Top-1:%.2f Top-5:%.2f\n'%(new_epoch, correct_top1, correct_topk))
        
        acc_test_log.write('Epoch:%d Loss:%.2f Top-1:%.2f Top-5:%.2f\n'%(epoch, test_loss, correct_top1, correct_topk))
        acc_test_log.flush()
    # save model
    # torch.save(vfl_lc, os.path.join(dir_save_model, f"{args.dataset}_saved_framework{setting_str}.pth"))
    torch.save(vfl_lc, os.path.join(dir_save_model, f"{args.dataset}_saved_framework{setting_str}.pth"), pickle_module=dill)


    if args.if_cluster_outputsA:
        outputsA_list = vfl_lc.outputs_a.detach().clone().cpu().numpy().tolist()
        labels_list = vfl_lc.labels_training_dataset.detach().clone().cpu().numpy().tolist()
        sp.cluster_outputs(args, outputsA_list, labels_list, setting_str)

if __name__ == '__main__':
    main()
