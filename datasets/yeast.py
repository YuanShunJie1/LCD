import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import sys
# from imblearn.over_sampling import SMOTE
# sys.path.insert(0, "./")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns


class YeastDataset(data.Dataset):

    def __init__(self, csv_path, train=True):
        """
        Args:
            csv_path (string): Path to the csv file.
        """
        csv_path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/yeast/yeast.data"
        
        self.train = train
        self.df = pd.read_csv(csv_path, sep=r'\s+')

        self.df.columns = ['Name', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Label']
        self.df = self.df.drop(columns=['Name'])
        # # 假设label是最后一列
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])

        y = self.df["Label"].values  # 标签
        x = self.df.drop(columns=["Label"]).values  # 特征（去除label列）

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

        sc = StandardScaler()

        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        self.train_data = x_train  # numpy array
        self.test_data = x_test

        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class YeastSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        base_dataset = YeastDataset(file_path)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.train_labels,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = YeastLabeled(file_path, train_labeled_idxs, train=True)
        train_unlabeled_dataset = YeastUnlabeled(file_path, train_unlabeled_idxs, train=True)
        train_complete_dataset = YeastLabeled(file_path, None, train=True)
        test_dataset = YeastLabeled(file_path, train=False)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_transforms(self):
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        _liver_dataset = YeastDataset(file_path, train)
        return _liver_dataset, _liver_dataset.train_labels

    def clip_one_party_data(self, x, half):
        x = x[:, :half]
        return x


class YeastLabeled(YeastDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(YeastLabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            self.train_labels = np.array(self.train_labels)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)


class YeastUnlabeled(YeastDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(YeastUnlabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            # self.train_labels = np.array(self.label_original)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)
        self.train_labels = np.array([-1 for i in range(len(self.train_labels))])


if __name__ == '__main__':
    path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/Yeast/yeast.data"
    dataset_setup = YeastSetup()
    train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset = dataset_setup.set_datasets_for_ssl(file_path=path, n_labeled=20, party_num=2)

    Yeast_train_set = YeastDataset(path, train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=Yeast_train_set,
        batch_size=128, shuffle=True,
        num_workers=0, pin_memory=True
    )
    Yeast_test_set = YeastDataset(path, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=Yeast_test_set,
        batch_size=128, shuffle=True,
        num_workers=0, pin_memory=True
    )
    print("len train loader:", len(train_loader))
    for batch_id, (data, target, index) in enumerate(train_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    print("\n\n test-->")
    print("len test loader:", len(test_loader))
    for batch_id, (data, target, index) in enumerate(test_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch target:", target)
        break

