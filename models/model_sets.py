"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init,  BasicBlock, weights_init_normal
import torch
# from models.mixtext import MixText


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet110(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[18, 18, 18], kernel_size=kernel_size, num_classes=num_classes)


def resnet56(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[9, 9, 9], kernel_size=kernel_size, num_classes=num_classes)



class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(14 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.reshape(-1, 14 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BottomModelForMnist(nn.Module):
    def __init__(self):
        super(BottomModelForMnist, self).__init__()
        self.mlpnet = MLPNet()

    def forward(self, x):
        x = self.mlpnet(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForMnist(nn.Module):
    def __init__(self):
        super(TopModelForMnist, self).__init__()
        self.fc1top = nn.Linear(10*2, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(10*2)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForMnist(nn.Module):
    def __init__(self,):
        super(FakeTopModelForMnist, self).__init__()
        self.fc1top = nn.Linear(10*2, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 5)
        self.bn0top = nn.BatchNorm1d(10*2)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(10*2, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(10*2)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForCifar10(nn.Module):
    def __init__(self,):
        super(FakeTopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(10*2, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 5)
        self.bn0top = nn.BatchNorm1d(10*2)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class BottomModelForCifar100(nn.Module):
    def __init__(self):
        super(BottomModelForCifar100, self).__init__()
        self.resnet20 = resnet20(num_classes=100)

    def forward(self, x):
        x = self.resnet20(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(100*2, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(100*2)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForCifar100(nn.Module):
    def __init__(self,):
        super(FakeTopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(100*2, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 50)
        self.bn0top = nn.BatchNorm1d(100*2)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


# class BottomModelForImageNet12(nn.Module):
#     def __init__(self):
#         super(BottomModelForImageNet12, self).__init__()
#         self.resnet56 = resnet56(num_classes=1000)

#     def forward(self, x):
#         x = self.resnet56(x)
#         x = F.normalize(x, dim=1)
#         return x

# class TopModelForImageNet12(nn.Module):
#     def __init__(self):
#         super(TopModelForImageNet12, self).__init__()
#         self.fc1top = nn.Linear(2000, 2000)
#         self.fc2top = nn.Linear(2000, 1000)
#         self.fc3top = nn.Linear(1000, 12)
#         self.bn0top = nn.BatchNorm1d(2000)
#         self.bn1top = nn.BatchNorm1d(2000)
#         self.bn2top = nn.BatchNorm1d(1000)

#         self.apply(weights_init)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         return F.log_softmax(x, dim=1)

# class FakeTopModelForImageNet12(nn.Module):
#     def __init__(self, output_times):
#         super(FakeTopModelForImageNet12, self).__init__()
#         self.fc1top = nn.Linear(2000, 2000)
#         self.fc2top = nn.Linear(2000, 1000)
#         self.fc3top = nn.Linear(1000, 6)
#         self.bn0top = nn.BatchNorm1d(2000)
#         self.bn1top = nn.BatchNorm1d(2000)
#         self.bn2top = nn.BatchNorm1d(1000)

#         self.apply(weights_init)

#     def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = output_bottom_models
#         x = self.fc1top(F.relu(self.bn0top(x)))
#         x = self.fc2top(F.relu(self.bn1top(x)))
#         x = self.fc3top(F.relu(self.bn2top(x)))
#         return F.log_softmax(x, dim=1)


# Yeast
# class TopModelForYeast(nn.Module):
#     def __init__(self,):
#         super(TopModelForYeast, self).__init__()
#         self.fc1_top = nn.Linear(20, 20)
#         self.bn0_top = nn.BatchNorm1d(20)
#         self.fc2_top = nn.Linear(20, 20)
#         self.bn1_top = nn.BatchNorm1d(20)
#         self.fc3_top = nn.Linear(20, 10)
#         self.bn2_top = nn.BatchNorm1d(20)
#         self.apply(weights_init_normal)

#     def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
#         output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
#         x = self.bn0_top(output_bottom_models)
#         x = F.relu(x)
#         x = self.fc1_top(x)
#         x = self.bn1_top(x)
#         x = F.relu(x)
#         x = self.fc2_top(x)
#         x = self.bn2_top(x)
#         x = F.relu(x)
#         x = self.fc3_top(x)
#         return F.log_softmax(x, dim=1)

class TopModelForYeast(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn1_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, 20)
        self.bn2_top = nn.BatchNorm1d(20)
        self.fc3_top = nn.Linear(20, 10)  # 输出层前不加 BN
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        x = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return F.log_softmax(x, dim=1)

class FakeTopModelForYeast(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn1_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, 20)
        self.bn2_top = nn.BatchNorm1d(20)
        self.fc3_top = nn.Linear(20, 5)  # 输出层前不加 BN
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        x = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return F.log_softmax(x, dim=1)

class BottomModelForYeast(nn.Module):
    def __init__(self,):
        super(BottomModelForYeast, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.apply(weights_init_normal)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.normalize(x, dim=1)
        return x


# Letter
class TopModelForLetter(nn.Module):
    def __init__(self,):
        super(TopModelForLetter, self).__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn0_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, 20)
        self.bn1_top = nn.BatchNorm1d(20)
        self.fc3_top = nn.Linear(20, 26)
        self.bn2_top = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0_top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return F.log_softmax(x, dim=1)

class FakeTopModelForLetter(nn.Module):
    def __init__(self,):
        super(FakeTopModelForLetter, self).__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn0_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, 20)
        self.bn1_top = nn.BatchNorm1d(20)
        self.fc3_top = nn.Linear(20, 13)
        self.bn2_top = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0_top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return F.log_softmax(x, dim=1)

class BottomModelForLetter(nn.Module):
    def __init__(self, ):
        super(BottomModelForLetter, self).__init__()
        self.fc1 = nn.Linear(8, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.normalize(x, dim=1)
        return x


    
    
    
    

class BasicBlock_L(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_L, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet_M(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_M, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden=False, return_activation=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        activation1 = out
        out = self.layer2(out)
        activation2 = out
        out = self.layer3(out)
        activation3 = out
        out = self.layer4(out)
        
        out = self.avgpool(out)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)

        if return_hidden:
            return out, hidden
        elif return_activation:  # for NAD
            return out, activation1, activation2, activation3
        else:
            return out

def resnet34(num_classes=12):
    return ResNet_M(BasicBlock_L, [3, 4, 6, 3], num_classes=num_classes)

class BottomModelForImageNet12(nn.Module):
    def __init__(self):
        super(BottomModelForImageNet12, self).__init__()
        self.resnet34 = resnet34(num_classes=128)

    def forward(self, x):
        x = self.resnet34(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForImageNet12(nn.Module):
    def __init__(self):
        super(TopModelForImageNet12, self).__init__()
        self.fc1top = nn.Linear(256, 512)
        self.fc2top = nn.Linear(512, 128)
        self.fc3top = nn.Linear(128, 12)
        self.bn0top = nn.BatchNorm1d(256)
        self.bn1top = nn.BatchNorm1d(512)
        self.bn2top = nn.BatchNorm1d(128)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForImageNet12(nn.Module):
    def __init__(self,):
        super(FakeTopModelForImageNet12, self).__init__()
        self.fc1top = nn.Linear(256, 512)
        self.fc2top = nn.Linear(512, 128)
        self.fc3top = nn.Linear(128, 6)
        self.bn0top = nn.BatchNorm1d(256)
        self.bn1top = nn.BatchNorm1d(512)
        self.bn2top = nn.BatchNorm1d(128)
        self.apply(weights_init)

    def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)




class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, half=None,is_adversary=True):
        name = self.dataset_name.lower()  # 转为小写匹配
        if name == 'mnist':
            return BottomModelForMnist()
        elif name == 'cifar10' or name == 'mycifar10':
            return BottomModelForCifar10()
        elif name == 'cifar100' or name == 'mycifar100':
            return BottomModelForCifar100()
        elif name == 'imagenet12':
            return BottomModelForImageNet12()
        elif name == 'yeast':
            return BottomModelForYeast()
        elif name == 'letter':
            return BottomModelForLetter()
        else:
            raise Exception(f'Unknown dataset name: {self.dataset_name}')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self):
        name = self.dataset_name.lower()
        if name == 'mnist':
            return TopModelForMnist()
        elif name == 'cifar10' or name == 'mycifar10':
            return TopModelForCifar10()
        elif name == 'cifar100' or name == 'mycifar100':
            return TopModelForCifar100()
        elif name == 'imagenet12':
            return TopModelForImageNet12()
        elif name == 'yeast':
            return TopModelForYeast()
        elif name == 'letter':
            return TopModelForLetter()
        else:
            raise Exception(f'Unknown dataset name: {self.dataset_name}')

class FakeTopModel:
    def __init__(self, dataset_name, output_times):
        self.dataset_name = dataset_name
        self.output_times = output_times
        
    def get_model(self):
        name = self.dataset_name.lower()
        if name == 'mnist':
            return FakeTopModelForMnist()
        elif name == 'cifar10' or name == 'mycifar10':
            return FakeTopModelForCifar10()
        elif name == 'cifar100' or name == 'mycifar100':
            return FakeTopModelForCifar100()
        elif name == 'imagenet12':
            return FakeTopModelForImageNet12()
        elif name == 'yeast':
            return FakeTopModelForYeast()
        elif name == 'letter':
            return FakeTopModelForLetter()
        else:
            raise Exception(f'Unknown dataset name: {self.dataset_name}')


def soft_cross_entropy(pred, soft_target, reduction='mean'):
    """
    计算 soft label 的交叉熵损失。
    
    参数:
        pred (Tensor): shape (B, C)，未归一化的 logits。
        soft_target (Tensor): shape (B, C)，soft label（伪标签，概率分布）。
        reduction (str): 'mean' | 'sum' | 'none'
    
    返回:
        标量损失值（如果 reduction='mean' 或 'sum'），或 shape (B,) 的向量（'none'）
    """
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(soft_target * log_probs, dim=1)  # shape: (B,)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # shape: (B,)

def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func, bottom_features=None):
    if bottom_features is None:
        loss = loss_func(output, batch_target)
    else:
        loss = loss_func(output, batch_target, bottom_features)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def update_top_model_one_batch_with_cae(optimizer, model, output, batch_target, loss_func,cae, bottom_features=None, num_class=10):

    # num_classes = cae.encoder[-1].out_features
    one_hot = F.one_hot(batch_target, num_classes=num_class).float().to(batch_target.device)
    pseudo_labels = cae.encoder(one_hot)  # 得到伪标签 (B, C)

    loss = soft_cross_entropy(output, pseudo_labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# self.mir_lambda
def update_top_model_one_batch_with_mir(optimizer, model, output, batch_target, loss_func, bottom_features=None, vib_module=None, mir_lambda=1e-3):
    # if bottom_features is None:
    #     loss = loss_func(output, batch_target)
    # else:
    #     loss = loss_func(output, batch_target, bottom_features)
    
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    if bottom_features is None or vib_module is None:
        # 无 passive 信息，正常训练
        loss = loss_func(output, batch_target)
    else:
        # 使用 VIB 机制处理 passive party 的嵌入
        mu, std = vib_module(bottom_features)

        # 拼接主动方与被动方的信息
        # joint_input = torch.cat([z, output], dim=1)

        # top model 输出预测
        # output = model(joint_input)

        # 交叉熵损失
        ce_loss = loss_func(output, batch_target)

        # KL 散度损失（信息瓶颈正则项）
        kl_loss = 0.5 * torch.mean(mu**2 + std**2 - torch.log(std**2 + 1e-8) - 1)

        loss = ce_loss + mir_lambda * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss




def update_fake_top_model_and_center_loss_one_batch(optimizer, model, output, batch_target, loss_func, criterion_cent, optimizer_centloss, bottom_features, weight_cent):

    loss_xent = loss_func(output, batch_target)
    loss_cent = criterion_cent(bottom_features, batch_target)
    
    loss_cent *= weight_cent
    loss = loss_xent + loss_cent
        
    optimizer.zero_grad()
    optimizer_centloss.zero_grad()
    
    loss.backward()
    optimizer.step()
    # by doing so, weight_cent would not impact on the learning of centers
    for param in criterion_cent.parameters():
        param.grad.data *= (1. / weight_cent)
        
    optimizer_centloss.step()
    
    # print("CE ", round(loss_xent.item(),4), " CL ", round(loss_cent.item(), 4))
    return loss

def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return




class VIBModule(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VIBModule, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2 * latent_dim)
        )

    def forward(self, x, num_sample=1):
        stats = self.encoder(x)
        mu = stats[:, :self.latent_dim]
        std = F.softplus(stats[:, self.latent_dim:] - 5)  # 保证 std 为正

        # if num_sample == 1:
        #     eps = torch.randn_like(std)
        #     z = mu + eps * std
        # else:
        #     raise NotImplementedError("仅支持单样本")
        return mu, std#, z


if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)


