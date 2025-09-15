# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch.nn import LayerNorm


batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head, device):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        # self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # self.prototype = nn.Parameter(self.head.state_dict()['weight'])

        scaling_tensor = torch.tensor([30.0]).to(device)
        self.scaling = torch.nn.Parameter(scaling_tensor)
        self.scaling.requires_grad_(False)

        # #只是获取形状用，不参与训练也不参与更新
        # weight_tensor = self.head.state_dict()['weight'].clone().detach()  # 克隆并分离权重以避免影响原始模型
        # weight_tensor = weight_tensor.to("cuda")  # 确保权重在 CUDA 设备上
        # self.prototype = torch.nn.Parameter(weight_tensor)
        # self.prototype.requires_grad = False



        
    def forward(self, x):
        out = self.base(x)  # 这里的out的形状与原型形状相同
        out = self.head(out)

        return out


class BaseHeadProjectSplit(nn.Module):
    def __init__(self, base, head, project_head,device=torch.device('cuda')):
        super(BaseHeadProjectSplit, self).__init__()

        self.base = base
        self.head = head
        self.project_head = project_head
        # self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        # self.prototype = nn.Parameter(self.head.state_dict()['weight'])

        scaling_tensor = torch.tensor([30.0], device=device)
        self.scaling = torch.nn.Parameter(scaling_tensor)
        self.scaling.requires_grad_(False)
        # #只是获取形状用，不参与训练也不参与更新
        # weight_tensor = self.head.state_dict()['weight'].clone().detach()  # 克隆并分离权重以避免影响原始模型
        # weight_tensor = weight_tensor.to("cuda")  # 确保权重在 CUDA 设备上
        # self.prototype = torch.nn.Parameter(weight_tensor)
        # self.prototype.requires_grad = False

    def forward(self, x, use_project_head=False):
        base_output = self.base(x)  # 这里的out的形状与原型形状相同
        if use_project_head:
            project_output = self.project_head(base_output)
            return project_output
        else:
            head_output = self.head(base_output)
            return head_output


###########################################################

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64 * 26, num_classes=6, conv_kernel_size=(1, 9),
                 pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/digit5.py
class Digit5CNN(nn.Module):
    def __init__(self):
        super(Digit5CNN, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())

        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return out


# https://github.com/FengHZ/KD3A/blob/master/model/amazon.py
class AmazonMLP(nn.Module):
    def __init__(self):
        super(AmazonMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 1000),
            # nn.BatchNorm1d(1000), 
            nn.ReLU(),
            nn.Linear(1000, 500),
            # nn.BatchNorm1d(500), 
            nn.ReLU(),
            nn.Linear(500, 100),
            # nn.BatchNorm1d(100), 
            nn.ReLU()
        )
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out


# # https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
# class FedAvgCNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features,
#                                32,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(32,
#                                64,
#                                kernel_size=5,
#                                padding=0,
#                                stride=1,
#                                bias=True)
#         self.fc1 = nn.Linear(dim, 512)
#         self.fc = nn.Linear(512, num_classes)

#         self.act = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.act(self.conv2(x))
#         x = self.maxpool(x)
#         x = torch.flatten(x, 1)
#         x = self.act(self.fc1(x))
#         x = self.fc(x)
#         return x

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        # base
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        # head
        self.fc = nn.Linear(512, num_classes)




    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out



class FedAvgCNN_WithBN(nn.Module):
    def __init__(self, in_features=3, num_classes=100, dim=1024):
        super().__init__()

        # 特征提取器部分
        self.feature_extractor = nn.Sequential(
            OrderedDict([
                # 第一个卷积块
                ('conv1', nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(32)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=(2, 2))),
                ('dropout1', nn.Dropout2d(0.1)),  # 添加2D Dropout

                # 第二个卷积块
                ('conv2', nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=(2, 2))),
                ('dropout2', nn.Dropout2d(0.1)),  # 添加2D Dropout
            ])
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            OrderedDict([
                # 第一个全连接层
                ('fc1', nn.Linear(dim, 512, bias=False)),
                ('bn3', nn.BatchNorm1d(512)),
                ('relu3', nn.ReLU(inplace=True)),
                ('dropout3', nn.Dropout(0.4)),  # 全连接层后添加较大dropout

                # 输出层
                ('fc2', nn.Linear(512, num_classes))
            ])

        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def produce_feature(self, x):
        """特征提取函数"""
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)  # 展平操作
        return x

    def forward(self, x):
        """前向传播函数"""
        # 特征提取
        x = self.produce_feature(x)

        # # 打印展平后的形状
        # print("Feature shape after flattening:", x.shape)

        # 分类
        x = self.classifier(x)
        return x




    # ====================================================================================================================

#proxy_model
class FedAvgCNNProxy(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        # base
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True)
        )
        # head
        self.fc = nn.Linear(1024, num_classes)




    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# ====================================================================================================================



class OnlineMLP_WithClassifier(nn.Module):
    """
    在线模型：分为 feature_extractor 和 classifier
    - feature_extractor：聚合部分
    - classifier：个性化或共享
    增加注意力机制
    """

    def __init__(self, in_features, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        input_dim = in_features

        # 更简单的特征提取网络（原版有3层+注意力，现在只有2层）
        for i, h in enumerate(hidden_dims):
            self.feature_extractor.add_module(f"fc{i + 1}", nn.Linear(input_dim, h))
            self.feature_extractor.add_module(f"bn{i + 1}", nn.BatchNorm1d(h))
            self.feature_extractor.add_module(f"relu{i + 1}", nn.ReLU())
            # 只在最后一层添加dropout（原版每层都有dropout）
            if i == len(hidden_dims) - 1:
                self.feature_extractor.add_module(f"dropout{i + 1}", nn.Dropout(dropout))
            input_dim = h

        # 简化的分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def produce_feature(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.produce_feature(x)
        out = self.classifier(features)
        return out.squeeze(1)


class OfflineMLP_WithClassifier(nn.Module):
    """
    离线模型：更强大，分为 feature_extractor 和 classifier
    - LayerNorm 替代 BN，增强稳定性
    增加注意力机制
    """

    def __init__(self, in_features, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        input_dim = in_features

        for i, h in enumerate(hidden_dims):
            self.feature_extractor.add_module(f"fc{i + 1}", nn.Linear(input_dim, h))
            self.feature_extractor.add_module(f"bn{i + 1}", nn.BatchNorm1d(h))
            self.feature_extractor.add_module(f"relu{i + 1}", nn.ReLU())
            if i == len(hidden_dims) - 1:
                self.feature_extractor.add_module(f"dropout{i + 1}", nn.Dropout(dropout))
            input_dim = h

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def produce_feature(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.produce_feature(x)
        out = self.classifier(features)
        return out.squeeze(1)


class FraudModel(nn.Module):
    """统一的欺诈检测模型架构，在线和离线模型共用"""

    def __init__(self, in_features, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential()
        input_dim = in_features

        for i, h in enumerate(hidden_dims):
            self.feature_extractor.add_module(f"fc{i + 1}", nn.Linear(input_dim, h))
            self.feature_extractor.add_module(f"bn{i + 1}", nn.BatchNorm1d(h))
            self.feature_extractor.add_module(f"relu{i + 1}", nn.ReLU())
            if i == len(hidden_dims) - 1:
                self.feature_extractor.add_module(f"dropout{i + 1}", nn.Dropout(dropout))
            input_dim = h

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def produce_feature(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.produce_feature(x)
        out = self.classifier(features)
        return out.squeeze(1)

    def get_layer_parameters(self):
        """按层返回参数组，用于分层训练"""
        layer_params = []

        # 特征提取层的参数
        for name, module in self.feature_extractor.named_children():
            if 'fc' in name or 'bn' in name:
                layer_params.append([module])

        # 分类器参数
        classifier_params = []
        for module in self.classifier:
            classifier_params.append(module)
        layer_params.append(classifier_params)

        return layer_params

# ====================================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
        self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LeNet(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_classes=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_classes)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ====================================================================================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out


# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.8,
                 padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input

#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output
