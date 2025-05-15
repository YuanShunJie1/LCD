import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch.nn.init as init

def random_initialize(model, init_type='kaiming_normal'):
    """
    对模型参数进行随机初始化（只对权重使用复杂初始化，bias 使用常数）。
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() < 2:
                # 跳过一维 weight（可能存在于 LayerNorm 等层）
                continue
            if init_type == 'xavier_uniform':
                init.xavier_uniform_(param)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(param)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(param, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(param, nonlinearity='relu')
            elif init_type == 'normal':
                init.normal_(param, mean=0.0, std=0.02)
            elif init_type == 'uniform':
                init.uniform_(param, a=-0.1, b=0.1)
            else:
                raise ValueError(f"不支持的初始化方式: {init_type}")
        elif 'bias' in name:
            init.constant_(param, 0.0)

def proccess_params(args):
    if args.dataset in ['mycifar10', 'cifar10', 'mnist']:
        args.num_class = 10
        args.feat_dim = 10
        args.num_new_classes = args.num_class // 2
        args.half = 16
        args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/cifar10'
        if args.dataset == 'mnist':
            args.half = 14
            args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/mnist'
        args.batch_size = 128
    elif args.dataset in ['mycifar100', 'cifar100']:
        args.num_class = 100
        args.feat_dim = 100
        args.num_new_classes = args.num_class // 2
        args.half = 16
        args.batch_size = 128
        args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/cifar100/cifar-100-python'
    elif args.dataset in ['imagenet12']:
        args.num_class = 12
        args.feat_dim = 128
        args.num_new_classes = args.num_class // 2
        args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/imagenet12'
        args.half = 112
        args.batch_size = 32
    elif args.dataset in ['yeast']:
        args.num_class = 10
        args.feat_dim = 10
        args.num_new_classes = args.num_class // 2
        args.lr = 0.005
        args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/yeast/yeast.data'
        args.half = 4
        args.batch_size = 128
    elif args.dataset in ['letter']:
        args.num_class = 26
        args.feat_dim = 10
        args.num_new_classes = args.num_class // 2
        args.lr = 0.01
        args.path_dataset = '/home/shunjie/codes/defend_label_inference/cs/Datasets/letter Recognition/letter-recognition.data'
        args.half = 8
        args.batch_size = 128
    return args    
            
def generate_log(args):
    dir_save_model = args.save_dir + f"/saved_models/{args.dataset}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)
    
    setting_str = "acc_"
    setting_str += "dataset="
    setting_str += str(args.dataset)
    setting_str += "_lr="
    setting_str += str(args.lr)
    if args.use_mal_optim:
        setting_str += "_"
        setting_str += "mal"
    else:
        setting_str += "_"
        setting_str += "normal"
    setting_str += "_"
    if args.dataset != 'Yahoo':
        setting_str += "half="
        setting_str += str(args.half)
    
    setting_str += "_batch_size="
    setting_str += str(args.batch_size)
    
    setting_str += "_num_new_classes="
    setting_str += str(args.num_new_classes)
    
    setting_str += "_gpu_id="
    setting_str += str(args.gpu_id)

    setting_str += "_feat_dim="
    setting_str += str(args.feat_dim)
    
    setting_str += "_weight_cent="
    setting_str += str(args.weight_cent)

    setting_str += ".txt"
    # dir_save_model = dir_save_model + setting_str
    # setting_str = os.path.join(dir_save_model, setting_str)
    
    print("settings:", setting_str)
    
    return setting_str, dir_save_model


def split_data(args, data):
    dataset = args.dataset.lower()
    if dataset in ['cifar10', 'cifar100', 'mycifar10', 'mycifar100', 'cinic10l']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif dataset == 'mnist':
        x_a = data[:, :, 0:args.half]
        x_b = data[:, :, args.half:28]        
    elif dataset == 'imagenet12':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:224]
    elif dataset == 'letter':
        x_b = data[:, 8:16]
        x_a = data[:, 0:8]
    elif dataset == 'yeast':
        x_b = data[:, 4:8]
        x_a = data[:, 0:4]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b

import sys
def myprint(args, epoch,batch_idx,num_iter,loss_framework):
    sys.stdout.write('\r')
    sys.stdout.write('%s | Epoch [%3d/%3d]  Iter[%3d/%3d]  CE-loss: %.4f\n'%(args.dataset, epoch, args.epochs, batch_idx+1, num_iter, loss_framework))
    sys.stdout.flush()



def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def test_per_epoch(args, test_loader, framework, k=5, loss_func_top_model=None, use_trandform=True, true2false={}):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    # count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().cuda()
            target = target.long().cuda()
            # set all sub-models to eval mode.
            framework.malicious_bottom_model_a.eval()
            framework.benign_bottom_model_b.eval()
            framework.top_model.eval()
            # run forward process of the whole framework
            x_a, x_b = split_data(args, data)
            output_tensor_bottom_model_a = framework.malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b = framework.benign_bottom_model_b(x_b)

            output_framework = framework.top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))
            test_loss += loss_func_top_model(output_framework, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch

        num_samples = len(test_loader.dataset)
        test_loss /= num_samples

    return test_loss, 100.00 * float(correct_top1) / num_samples, 100.00 * float(correct_topk) / num_samples

def test_per_epoch_models(args, test_loader, models, k=5, loss_func_top_model=None):
    malicious_bottom_model_a, benign_bottom_model_b, top_model = models
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    # count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().cuda()
            target = target.long().cuda()
            # set all sub-models to eval mode.
            malicious_bottom_model_a.eval()
            benign_bottom_model_b.eval()
            
            top_model.eval()
            # run forward process of the whole 
            x_a, x_b = split_data(args, data)
            output_tensor_bottom_model_a =  malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b =  benign_bottom_model_b(x_b)

            output_framework = top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))
            # sum up batch loss
            test_loss += loss_func_top_model(output_framework, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch

        num_samples = len(test_loader.dataset)
        test_loss /= num_samples

    return test_loss, 100.00 * float(correct_top1) / num_samples, 100.00 * float(correct_topk) / num_samples


def cluster_outputs(args, outputsA_list, labels_list, setting_str):
    # outputsA_list = model.outputs_a.detach().clone().cpu().numpy().tolist()
    # labels_list = model.labels_training_dataset.detach().clone().cpu().numpy().tolist()
    # plot TSNE cluster result
    if args.num_class != 10:
        return 
    outputsA_pca_tsne = TSNE()
    outputsA_list = np.array(outputsA_list)
    outputsA_pca_tsne.fit_transform(outputsA_list)
    df_outputsA_pca_tsne = pd.DataFrame(outputsA_pca_tsne.embedding_, index=labels_list)
    # plot the TSNE result
    colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'grey', 'orange', 'pink']
    # get num_classes
    # dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    # num_classes = dataset_setup.num_classes
    num_classes = args.num_new_classes
    for i in range(num_classes):
        plt.scatter(df_outputsA_pca_tsne.loc[i][0], df_outputsA_pca_tsne.loc[i][1], color=colors[i], marker='.')
    plt.title('VFL OutputsA TSNE' + setting_str)
    # plt.show()
    dir_save_tsne_pic = args.save_dir + f"/csv_files/{args.dataset}_csv_files"
    if not os.path.exists(dir_save_tsne_pic):
        os.makedirs(dir_save_tsne_pic)
    df_outputsA_pca_tsne.to_csv(
        dir_save_tsne_pic + f"/{args.dataset}_outputs_a_tsne{setting_str}.csv")
    plt.savefig(os.path.join(dir_save_tsne_pic, f"{args.dataset}_Resnet_VFL_OutputsA_TSNE{setting_str}.png"))
    plt.close()


    # # 初始化伪标签
    # def initialize_fake_labels_and_features(self, labels):
    #     num_classes = len(set(labels))
    #     num_new_classes = args.num_new_classes

    #     true_labels = list(range(0, num_classes))
    #     fake_labels = list(range(0, num_new_classes))

    #     c = num_classes // num_new_classes
    #     r = num_classes - c * num_new_classes

    #     len_box = [c for i in range(num_new_classes)]
    #     for i in range(r):
    #         len_box[i] = len_box[i] + 1
        
    #     random.shuffle(true_labels)
        
    #     start = 0
    #     for i in range(num_new_classes):
    #         self.false2true[i] = true_labels[start:start+len_box[i]]
    #         start = start+len_box[i]
        
    #     for newlabel in self.false2true:
    #         for true_label in self.false2true[newlabel]:
    #             self.true2false[true_label] = newlabel
        
    #     for c in range(num_classes):
    #         indices = np.where(c == np.array(labels))[0].tolist()
    #         for idx in indices:
    #             self.idx2falselabels[idx] = self.true2false[c]
        
    #     print('Fake and ground-truth labels info:\n')
    #     print('True labels to fake labels\n')
    #     print(self.true2false)
    #     print('Fake labels to true labels\n')
    #     print(self.false2true)
    #     print('\n')