# from datasets import bc_idc, cifar10, cifar100, mycifar10, mycifar100, cinic10, yahoo, tiny_image_net, criteo, breast_cancer_wisconsin, mnist
from datasets import cifar10, cifar100, mnist, yeast, letter, imagenet12

import torchvision.datasets as datasets

def get_dataset_by_name(dataset_name):
    dict_dataset = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
        'imagenet12': imagenet12.ImageNet12Dataset,
        'yeast': yeast.YeastDataset,
        'letter': letter.LetterDataset,
    }
    dataset = dict_dataset[dataset_name]
    return dataset

def get_datasets_for_ssl(dataset_name, file_path, n_labeled, party_num=None):
    dataset_setup = get_dataset_setup_by_name(dataset_name)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = \
        dataset_setup.set_datasets_for_ssl(file_path, n_labeled, party_num)
    return train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset

def get_dataset_setup_by_name(dataset_name):
    dict_dataset_setup = {
        'cifar10': cifar10.Cifar10Setup(),
        'cifar100': cifar100.Cifar100Setup(),
        'mnist': mnist.MnistSetup(),
        'imagenet12': imagenet12.ImageNet12Setup(),
        'yeast': yeast.YeastSetup(),
        'letter': letter.LetterSetup(),
    }
    dataset_setup = dict_dataset_setup[dataset_name]
    return dataset_setup


# model_completion_yeast_saved_framework_vfl_framework_lr=0.05_mal_gc-preserved_percent=0.9_half=4_batch_size=128.pth_layer=1_func=ReLU_bn=True_nlabeled=400.txt
# model_completion_yeast_saved_framework_vfl_framework_lr=0.05_mal_gc-preserved_percent=0.9_half=4_batch_size=128.pth_layer=1_func=ReLU_bn=True_nlabeled=400.txt