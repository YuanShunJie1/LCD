import os
import glob
import sys
# sys.path.insert(0, "./")

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split, image_format_2_rgb



class ImageNet12Dataset(datasets.ImageFolder):
    def __init__(self, root,transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label  # 保持你原先返回 index 的格式

class SubsetImageNet12(Dataset):
    def __init__(self, base_dataset, indices, unlabeled=False):
        self.dataset = base_dataset
        self.indices = indices
        self.unlabeled = unlabeled

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        if self.unlabeled:
            label = -1
        return img, label

    def __len__(self):
        return len(self.indices)

class ImageNet12Setup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 12
        self.size_bottom_out = 128  # 根据你数据集具体情况设置

    def get_normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def get_transforms(self):
        normalize = self.get_normalize_transform()
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        split_dir = 'train' if train else 'val'
        transforms_ = self.get_transforms()
        dataset = ImageNet12Dataset(os.path.join(file_path, split_dir), transform=transforms_)
        return dataset, [label for (_, label) in dataset.samples]

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        transforms_ = self.get_transforms()
        base_dataset = ImageNet12Dataset(os.path.join(file_path, 'train'), transform=transforms_)
        labels = [label for _, label in base_dataset.samples]
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(labels, int(n_labeled / self.num_classes), self.num_classes)

        train_labeled_dataset = SubsetImageNet12(base_dataset, train_labeled_idxs)
        train_unlabeled_dataset = SubsetImageNet12(base_dataset, train_unlabeled_idxs, unlabeled=True)
        train_complete_dataset = base_dataset
        test_dataset = ImageNet12Dataset(os.path.join(file_path, 'val'), transform=transforms_)

        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def clip_one_party_data(self, x, half):
        return x[:, :, :, :half]



if __name__ == '__main__':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    # augmentation = transforms.RandomApply([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.RandomResizedCrop(64)], p=.8)

    train_transform = transforms.Compose([
        # transforms.Lambda(image_format_2_rgb),
        # augmentation,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_imagenet])

    test_transform = transforms.Compose([
        # transforms.Lambda(image_format_2_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_imagenet])

    dataset_train = ImageNet12Dataset(root='/home/shunjie/codes/defend_label_inference/cs/Datasets/imagenet12/train', transform=train_transform)
    dataset_test = ImageNet12Dataset(root='/home/shunjie/codes/defend_label_inference/cs/Datasets/imagenet12/val', transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True
    )
    print("len train loader:", len(train_loader))
    for batch_id, (data, target, index) in enumerate(train_loader):
        print("batch_id:", batch_id)
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    print("\n\n test-->")
    print("len test loader:", len(test_loader))
    for batch_id, (data, target, index) in enumerate(test_loader):
        print("batch_id:", batch_id)
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    for data, target, index in test_loader:
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
