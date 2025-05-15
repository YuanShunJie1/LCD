"""
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init,  BasicBlock
import torch
from models.mixtext import MixText
from torch.autograd import Variable



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
    
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=14):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)



class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet18 = ResNet18(num_classes=128)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, 10)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
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
    def __init__(self, output_times):
        super(FakeTopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, output_times)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)



class BottomModelForCinic10(nn.Module):
    def __init__(self):
        super(BottomModelForCinic10, self).__init__()
        self.resnet18 = ResNet18(num_classes=128)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCinic10(nn.Module):
    def __init__(self):
        super(TopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, 10)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForCinic10(nn.Module):
    def __init__(self, output_times):
        super(FakeTopModelForCinic10, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, output_times)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
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
        self.resnet18 = ResNet18(num_classes=128)

    def forward(self, x):
        x = self.resnet18(x)
        x = F.normalize(x, dim=1)
        return x

class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, 100)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
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
    def __init__(self, output_times):
        super(FakeTopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(128*2, 128)
        self.fc2top = nn.Linear(128, 256)
        self.fc3top = nn.Linear(256, 512)
        self.fc4top = nn.Linear(512, output_times)
        self.bn0top = nn.BatchNorm1d(128*2)
        self.bn1top = nn.BatchNorm1d(128)
        self.bn2top = nn.BatchNorm1d(256)
        self.bn3top = nn.BatchNorm1d(512)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)



class BottomModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(BottomModelForTinyImageNet, self).__init__()
        self.resnet50 = ResNet50(num_classes=200)
    
    def forward(self, x):
        x = self.resnet50(x)
        x = F.normalize(x, dim=1)
        return x


class TopModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(TopModelForTinyImageNet, self).__init__()
        self.fc1top = nn.Linear(400, 400)
        self.fc2top = nn.Linear(400, 200)
        self.fc3top = nn.Linear(200, 200)
        self.bn0top = nn.BatchNorm1d(400)
        self.bn1top = nn.BatchNorm1d(400)
        self.bn2top = nn.BatchNorm1d(200)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)

class FakeTopModelForTinyImageNet(nn.Module):
    def __init__(self,output_times):
        super(FakeTopModelForTinyImageNet, self).__init__()
        self.fc1top = nn.Linear(400, 400)
        self.fc2top = nn.Linear(400, 200)
        self.fc3top = nn.Linear(200, output_times)
        self.bn0top = nn.BatchNorm1d(400)
        self.bn1top = nn.BatchNorm1d(400)
        self.bn2top = nn.BatchNorm1d(200)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)



class BottomModelForImageNet(nn.Module):
    def __init__(self):
        super(BottomModelForImageNet, self).__init__()
        self.resnet50 = resnet50(num_classes=1000)

    def forward(self, x):
        x = self.resnet50(x)
        x = F.normalize(x, dim=1)
        return x


class TopModelForImageNet(nn.Module):
    def __init__(self):
        super(TopModelForImageNet, self).__init__()
        self.fc1top = nn.Linear(2000, 2000)
        self.fc2top = nn.Linear(2000, 1000)
        self.fc3top = nn.Linear(1000, 1000)
        self.bn0top = nn.BatchNorm1d(2000)
        self.bn1top = nn.BatchNorm1d(2000)
        self.bn2top = nn.BatchNorm1d(1000)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)


class FakeTopModelForImageNet(nn.Module):
    def __init__(self, output_times):
        super(FakeTopModelForImageNet, self).__init__()
        self.fc1top = nn.Linear(2000, 2000)
        self.fc2top = nn.Linear(2000, 1000)
        self.fc3top = nn.Linear(1000, output_times)
        self.bn0top = nn.BatchNorm1d(2000)
        self.bn1top = nn.BatchNorm1d(2000)
        self.bn2top = nn.BatchNorm1d(1000)

        self.apply(weights_init)

    def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)


class TopModelForYahoo(nn.Module):

    def __init__(self):
        super(TopModelForYahoo, self).__init__()
        self.fc1_top = nn.Linear(20, 10)
        self.fc2_top = nn.Linear(10, 10)
        self.fc3_top = nn.Linear(10, 10)
        self.fc4_top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(10)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        x = self.bn3top(x)
        x = F.relu(x)
        x = self.fc4_top(x)

        return x


class FakeTopModelForYahoo(nn.Module):

    def __init__(self, output_times):
        super(FakeTopModelForYahoo, self).__init__()
        self.fc1_top = nn.Linear(20, 10)
        self.fc2_top = nn.Linear(10, 10)
        self.fc3_top = nn.Linear(10, 10)
        self.fc4_top = nn.Linear(10, output_times)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(10)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        self.apply(weights_init)

    def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = self.bn2top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        x = self.bn3top(x)
        x = F.relu(x)
        x = self.fc4_top(x)

        return x



class BottomModelForYahoo(nn.Module):

    def __init__(self, n_labels):
        super(BottomModelForYahoo, self).__init__()
        self.mixtext_model = MixText(n_labels, True)

    def forward(self, x):
        x = self.mixtext_model(x)
        x = F.normalize(x, dim=1)
        return x


D_ = 2 ** 13


class TopModelForCriteo(nn.Module):

    def __init__(self):
        super(TopModelForCriteo, self).__init__()
        self.fc1_top = nn.Linear(8, 8)
        self.fc2_top = nn.Linear(8, 4)
        self.fc3_top = nn.Linear(4, 2)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = F.relu(x)
        x = self.fc1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return x


class FakeTopModelForCriteo(nn.Module):
    def __init__(self, output_times):
        super(FakeTopModelForCriteo, self).__init__()
        self.fc1_top = nn.Linear(8, 8)
        self.fc2_top = nn.Linear(8, 4)
        self.fc3_top = nn.Linear(4, output_times)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = F.relu(x)
        x = self.fc1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = self.fc3_top(x)
        return x



class BottomModelForCriteo(nn.Module):

    def __init__(self, half=14, is_adversary=False):
        super(BottomModelForCriteo, self).__init__()
        if not is_adversary:
            half = D_ - half
        self.fc1 = nn.Linear(half, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


# class TopModelForBcw(nn.Module):
#     def __init__(self):
#         super(TopModelForBcw, self).__init__()
#         self.fc1_top = nn.Linear(4, 4)
#         self.bn0_top = nn.BatchNorm1d(4)
#         self.fc2_top = nn.Linear(4, 4)
#         self.bn1_top = nn.BatchNorm1d(4)
#         self.fc3_top = nn.Linear(4, 4)
#         self.bn2_top = nn.BatchNorm1d(4)
#         self.fc4_top = nn.Linear(4, 4)
#         self.bn3_top = nn.BatchNorm1d(4)
#         self.apply(weights_init)

#     def forward(self,input_tensor_top_model_a, input_tensor_top_model_b):
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
#         x = self.bn3_top(x)
#         x = F.relu(x)        
#         x = self.fc4_top(x)
#         return x


class TopModelForBcw(nn.Module):
    def __init__(self,):
        super(TopModelForBcw, self).__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn0_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, 20)
        self.bn1_top = nn.BatchNorm1d(20)
        self.fc3_top = nn.Linear(20, 2)
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
        return x


class FakeTopModelForBcw(nn.Module):
    def __init__(self, output_times):
        super(FakeTopModelForBcw, self).__init__()
        self.fc1_top = nn.Linear(20, 20)
        self.bn0_top = nn.BatchNorm1d(20)
        self.fc2_top = nn.Linear(20, int(2*output_times))
        self.bn1_top = nn.BatchNorm1d(20)
        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = self.bn0_top(output_bottom_models)
        x = F.relu(x)
        x = self.fc1_top(x)
        x = self.bn1_top(x)
        x = F.relu(x)
        x = self.fc2_top(x)
        return x



class BottomModelForBcw(nn.Module):
    def __init__(self, half=14, is_adversary=False):
        super(BottomModelForBcw, self).__init__()
        if not is_adversary:
            half = 28 - half
        self.fc1 = nn.Linear(half, 20)
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
        return x


class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, half, is_adversary, n_labels=10):
        if self.dataset_name == 'ImageNet':
            return BottomModelForImageNet()
        elif self.dataset_name == 'CIFAR10' or self.dataset_name == 'MYCIFAR10':
            return BottomModelForCifar10()
        elif self.dataset_name == 'CIFAR100' or self.dataset_name == 'MYCIFAR100':
            return BottomModelForCifar100()
        elif self.dataset_name == 'TinyImageNet':
            return BottomModelForTinyImageNet()
        elif self.dataset_name == 'CINIC10L':
            return BottomModelForCinic10()
        elif self.dataset_name == 'Yahoo':
            return BottomModelForYahoo(n_labels)
        elif self.dataset_name == 'Criteo':
            return BottomModelForCriteo(half, is_adversary)
        elif self.dataset_name == 'BCW':
            return BottomModelForBcw(half, is_adversary)
        else:
            raise Exception('Unknown dataset name!')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self):
        if self.dataset_name == 'ImageNet':
            return TopModelForImageNet()
        elif self.dataset_name == 'CIFAR10' or self.dataset_name == 'MYCIFAR10':
            return TopModelForCifar10()
        elif self.dataset_name == 'CIFAR100' or self.dataset_name == 'MYCIFAR100':
            return TopModelForCifar100()
        elif self.dataset_name == 'TinyImageNet':
            return TopModelForTinyImageNet()
        elif self.dataset_name == 'CINIC10L':
            return TopModelForCinic10()
        elif self.dataset_name == 'Yahoo':
            return TopModelForYahoo()
        elif self.dataset_name == 'Criteo':
            return TopModelForCriteo()
        elif self.dataset_name == 'BCW':
            return TopModelForBcw()
        else:
            raise Exception('Unknown dataset name!')

class FakeTopModel:
    def __init__(self, dataset_name, output_times):
        self.dataset_name = dataset_name
        self.output_times = output_times
    def get_model(self):
        if self.dataset_name == 'ImageNet':
            return FakeTopModelForImageNet(self.output_times)
        elif self.dataset_name == 'CIFAR10' or self.dataset_name == 'MYCIFAR10':
            return FakeTopModelForCifar10(self.output_times)
        elif self.dataset_name == 'CIFAR100' or self.dataset_name == 'MYCIFAR100':
            return FakeTopModelForCifar100(self.output_times)
        elif self.dataset_name == 'TinyImageNet':
            return FakeTopModelForTinyImageNet(self.output_times)
        elif self.dataset_name == 'CINIC10L':
            return FakeTopModelForCinic10(self.output_times)
        elif self.dataset_name == 'Yahoo':
            return FakeTopModelForYahoo(self.output_times)
        elif self.dataset_name == 'Criteo':
            return FakeTopModelForCriteo(self.output_times)
        elif self.dataset_name == 'BCW':
            return FakeTopModelForBcw(self.output_times)
        else:
            raise Exception('Unknown dataset name!')


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func, bottom_embeddings=None):
    if bottom_embeddings is None:
        loss = loss_func(output, batch_target)
    else:
        loss = loss_func(output, batch_target, bottom_embeddings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)
