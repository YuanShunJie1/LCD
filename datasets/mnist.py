"""
Thanks: MixMatch-Pytorch
"""
import numpy as np
import torchvision
from torchvision import transforms, datasets

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split
from PIL import Image


class MnistSetup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        transforms_ = self.get_transforms()
        base_dataset = torchvision.datasets.MNIST(file_path, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.targets, int(n_labeled / self.num_classes), self.num_classes)
        train_labeled_dataset = MNISTLabeled(file_path, train_labeled_idxs, train=True, transform=transforms_)
        train_unlabeled_dataset = MNISTUnlabeled(file_path, train_unlabeled_idxs, train=True, transform=transforms_)
        train_complete_dataset = MNISTLabeled(file_path, None, train=True, transform=transforms_)
        test_dataset = MNISTLabeled(file_path, train=False, transform=transforms_, download=True)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_transforms(self):
        transform_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        transforms_ = self.get_transforms()
        _cifar10_dataset = datasets.MNIST(file_path, train, transform=transforms_, download=True)

        labels = _cifar10_dataset.targets
        labels_list = labels.tolist()

        return _cifar10_dataset, labels_list

    def clip_one_party_data(self, x, half):
        x = x[:, :, :, :half]
        return x

mnist_mean = 0.1307
mnist_std  = 0.3081

def normalise(x, mean=mnist_mean, std=mnist_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


class MNISTLabeled(torchvision.datasets.MNIST):

    def __init__(self, root, indexs=None, train=True, transform=None, target_transform=None, download=True):
        super(MNISTLabeled, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        # self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTUnlabeled(MNISTLabeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNISTUnlabeled, self).__init__(root, indexs, train=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
