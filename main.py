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

# !/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from torch.nn import Sequential

from flcore.servers.serverNew import NewServer
from flcore.servers.serveravg import FedAvg
from flcore.servers.serveravgpush import AvgPush
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serveravgDBE import FedAvgDBE
from flcore.servers.serverNH import FedNH
from flcore.servers.serverproxy import ProxyFL
from flcore.servers.serverppd import PPDFL
from flcore.servers.serverDFedAvg import DFedAvg
from flcore.servers.serverDFedAvgM import DFedAvgM
from flcore.servers.serveras import FedAS
from flcore.servers.serverMyMethod import MyMethod
from flcore.servers.serverMyMethod_Xiaorong import MyMethod_Xiaorong

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from system.utils.result_utils import average_data
from system.utils.result_utils import results_store
from system.utils.mem_utils import MemReporter
from system.utils.data_utils import read_client_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

torch.manual_seed(233)  # 改随机种子

# hyper-params for Text tasks
vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    offline_model_str = args.offline_model
    proxy_model_str = args.proxy_model

    sample_data = read_client_data(dataset=args.dataset, idx=0, is_train=True)
    x, y = sample_data[0]
    in_features = x.shape[0]
    print("检测到输入特征数:", in_features)

    if args.algorithm == "ProxyFL" or args.algorithm == "PPDFL":
        args.use_proxy = True

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":  # non-convex
            # if "mnist" in args.dataset:
            #     args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            # elif "Cifar10" in args.dataset:
            #     args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            # elif "omniglot" in args.dataset:
            #     args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # elif "Digit5" in args.dataset:
            #     args.model = Digit5CNN().to(args.device)
            # else:
            #     args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

            if "mnist" in args.dataset:
                args.model = FedAvgCNN_WithBN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN_WithBN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN_WithBN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            elif "Cifar100" in args.dataset:
                args.model = FedAvgCNN_WithBN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN_WithBN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)


        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)


        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                   output_size=args.num_classes,
                                                   num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                   attention_dropout=0,
                                                   embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2,
                                          num_classes=args.num_classes).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
        elif model_str == "MLP":
            if args.dataset == 'FraudDetection':
                # args.model = OnlineMLP_WithClassifier(in_features=in_features, hidden_dims=[256, 128])
                args.model = FraudModel(in_features=in_features, hidden_dims=[256, 128])
                if offline_model_str == "MLP":
                 # args.offline_model = OfflineMLP_WithClassifier(in_features=in_features, hidden_dims=[256, 128])
                 args.offline_model = FraudModel(in_features=in_features, hidden_dims=[256, 128])


        else:
            raise NotImplementedError

        # Generate args.proxy_model
        if args.use_proxy == True:
            if proxy_model_str == "mlr":  # convex
                if "mnist" in args.dataset:
                    args.proxy_model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.proxy_model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
                else:
                    args.proxy_model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "cnn":  # non-convex
                if "mnist" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
                elif "omniglot" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                elif "Digit5" in args.dataset:
                    args.proxy_model = Digit5CNN().to(args.device)
                else:
                    args.proxy_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

            elif proxy_model_str == "dnn":  # non-convex
                if "mnist" in args.dataset:
                    args.proxy_model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.proxy_model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
                else:
                    args.proxy_model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "resnet":
                args.proxy_model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(
                    args.device)

            elif proxy_model_str == "resnet10":
                args.proxy_model = resnet10(num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "resnet34":
                args.proxy_model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(
                    args.device)

            elif proxy_model_str == "alexnet":
                args.proxy_model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "googlenet":
                args.proxy_model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                                num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "mobilenet_v2":
                args.proxy_model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "lstm":
                args.proxy_model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                    args.device)

            elif proxy_model_str == "bilstm":
                args.proxy_model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                                             output_size=args.num_classes,
                                                             num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                                             attention_dropout=0,
                                                             embedding_length=emb_dim).to(args.device)

            elif proxy_model_str == "fastText":
                args.proxy_model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
                    args.device)

            elif proxy_model_str == "TextCNN":
                args.proxy_model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                                           num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "Transformer":
                args.proxy_model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim,
                                                    nlayers=2,
                                                    num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "AmazonMLP":
                args.proxy_model = AmazonMLP().to(args.device)

            elif proxy_model_str == "harcnn":
                if args.dataset == 'har':
                    args.proxy_model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                              pool_kernel_size=(1, 2)).to(args.device)
                elif args.dataset == 'pamap':
                    args.proxy_model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                              pool_kernel_size=(1, 2)).to(args.device)

            else:
                raise NotImplementedError
            print(args.proxy_model)

        # select algorithm
        if args.algorithm == "FedAvg" and args.model.__class__.__name__ == "FedAvgCNN":
            args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
            args.model.fc = nn.Identity()  # 清空全连接层
            args.model = BaseHeadSplit(args.model, args.head,args.device)  # 将模型分为head和base两部分
            server = FedAvg(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）
        elif args.algorithm == "FedAvg" and args.model.__class__.__name__ == "FedAvgCNN_WithBN":
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head, args.device)
            # server = FedROD(args, i)

            # 获取最后一个全连接层（分类器的输出层）---------------针对模型FedAvgCNN_WithBN的改动
            args.head = copy.deepcopy(args.model.classifier[-1])  # 获取fc2层
            # 移除原始分类器的最后一层
            args.model.classifier = args.model.classifier[:-1]
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head, args.device)
            # server = MOON(args, i)

            # 获取最后一个全连接层（分类器的输出层）---------------针对模型FedAvgCNN_WithBN的改动
            args.head = copy.deepcopy(args.model.classifier[-1])  # 获取fc2层
            # 移除原始分类器的最后一层
            args.model.classifier = args.model.classifier[:-1]
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = MOON(args, i)


        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.classifier[-1])  # 获取fc2层
            # 移除原始分类器的最后一层
            args.model.classifier = args.model.classifier[:-1]
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedAvgDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvgDBE(args, i)

        elif args.algorithm == "FedNH":
            in_features = args.model.fc.in_features
            out_features = args.model.fc.out_features
            args.head = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(args.device)
            # 512*10
            m, n = args.head.weight.shape
            args.head.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(args.device)
            # args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
            args.model.fc = nn.Identity()  # 清空全连接层
            args.model = BaseHeadSplit(args.model, args.head, args.device)  # 将模型分为head和base两部分
            args.model.head.state_dict()['weight'].requires_grad = False
            server = FedNH(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）

        elif args.algorithm == "AvgPush":
            args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
            args.model.fc = nn.Identity()  # 清空全连接层
            args.model = BaseHeadSplit(args.model, args.head,args.device)  # 将模型分为head和base两部分
            server = AvgPush(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）

        elif args.algorithm == "ProxyFL":
            args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
            args.model.fc = nn.Identity()  # 清空全连接层
            args.model = BaseHeadSplit(args.model, args.head,args.device)  # 将模型分为head和base两部分
            server = ProxyFL(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）

        elif args.algorithm == "PPDFL":
            args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
            args.model.fc = nn.Identity()  # 清空全连接层
            args.model = BaseHeadSplit(args.model, args.head, args.device)  # 将模型分为head和base两部分
            for param in args.proxy_model.fc.parameters():
                param.requires_grad = False
            server = PPDFL(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）

        # elif args.algorithm == "DFedDC":
        #     args.head = copy.deepcopy(args.model.fc)
        #     args.model.fc = nn.Identity()
        #     args.model = BaseHeadSplit(args.model, args.head, args.device)
        #     server = DFedDC(args, i)

        elif args.algorithm == "DFedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = DFedAvg(args, i)

        elif args.algorithm == "DFedAvgM":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = DFedAvgM(args, i)

        elif args.algorithm == 'FedAS':

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head, args.device)
            server = FedAS(args, i)
            # args.head = copy.deepcopy(args.model.classifier[-1])  # 获取fc2层
            # # 移除原始分类器的最后一层
            # args.model.classifier = args.model.classifier[:-1]
            # # args.head = copy.deepcopy(args.model.fc)
            # # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head, args.device)
            # server = FedAS(args, i)


        # elif args.algorithm == "MyMethod":
        #     in_features = args.model.fc.in_features
        #     out_features = args.model.fc.out_features
        #     args.head = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(args.device)
        #     args.project_head = nn.Linear(in_features=in_features, out_features=64, bias=False).to(args.device)
        #     # 512*10
        #
        #     torch.nn.init.orthogonal_(args.head.weight)  # 直接在权重上应用正交初始化
        #
        #     # m, n = args.head.weight.shape
        #     # args.head.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(args.device)
        #     # args.head = copy.deepcopy(args.model.fc)  # 将模型的全连接层复制到head
        #     args.model.fc = nn.Identity()  # 清空全连接层
        #     args.model = BaseHeadProjectSplit(args.model, args.head, args.project_head,
        #                                       args.device)  # 将模型分为head和base两部分
        #     args.model.head.state_dict()['weight'].requires_grad = False
        #     server = NewServer(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端

        elif args.algorithm == "MyMethod":
            server = MyMethod(args, i)

        elif args.algorithm == "MyMethod_Xiaorong":
            server = MyMethod_Xiaorong(args, i)

        else:
            raise NotImplementedError
        print(args.model)

        server.train()  # 训练



        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    # results_store(server)
    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="FraudDetection")
    parser.add_argument('-nb', "--num_classes", type=int, default=2)
    parser.add_argument('-m', "--model", type=str, default="MLP")
    parser.add_argument('-offm', "--offline_model", type=str, default="MLP")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)

    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="MyMethod")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    parser.add_argument('-opt', "--optimizer", type=str, default='SGD',
                        help="Optimizer")  # ----------新加的-----------
    parser.add_argument('-sgd_momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('-sgd_weight_decay', default=1e-4, type=float, help='sgd weight decay')

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedNH
    parser.add_argument('--FedNH_smoothing', default=0.9, type=float, help='moving average parameters')
    parser.add_argument('--FedNH_server_adv_prototype_agg', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH server adv agg')
    parser.add_argument('--FedNH_client_adv_prototype_agg', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH client adv agg')
    parser.add_argument('--no_norm', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Use group/batch norm or not')
    parser.add_argument('--use_sam', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Use SAM optimizer')
    parser.add_argument('--FedNH_head_init', default="orthogonal", type=str, help='FedNH head init')
    parser.add_argument('--FedNH_lr_scheduler', default="stepwise", type=str, help='FedNH learning rate scheduler')
    parser.add_argument('-use_proxy', type=bool, default=False, help="use proxy_model or not")

    # ProxyFL
    parser.add_argument('-proxy_m', "--proxy_model", type=str, default="cnn")

    # PPDFL
    parser.add_argument('--seq_length', type=int, default=10, help='DGFL sequence length.')
    parser.add_argument('--epsilon', type=float, default=1, help="epsilon-greedy parameter.")
    parser.add_argument('--top_k', type=float, default=0.2, help="proportion of recipients to be selected.")

    # DFedAvgM
    parser.add_argument('--Mbeta', type=float, default=0.9, help="parameter beta for momentum.")
    parser.add_argument('--itr_K', type=int, default=2, help="local training iteration times.")

    # Byzantine
    parser.add_argument('--bzt', type=bool, default=False, help="whether choose byzantine scenario.")
    parser.add_argument('--malicious_ratio', type=float, default=0.2, help="the ratio of malicious.")
    parser.add_argument('--poison_ratio', type=float, default=0.5, help="the ratio of poison data in malicious.")
    # 每次重新生成数据集都要改下面这个参数
    parser.add_argument('--malicious_ids', type=list, default=[-1], help="malicious indexes.")

    # MyMethod
    parser.add_argument('--patience', type=int, default=40, help="patience for early stop")
    parser.add_argument('--early_stop_mode', type=str, default="max", help="type of early stop mode:max/min")
    parser.add_argument('--min_delta', type=float, default=0.0001, help="min of the progress")
    parser.add_argument('--max_layers', type=int, default=3, help='最大层级数')
    parser.add_argument('--min_clients_per_cluster', type=int, default=2, help='每个聚类最小客户端数')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='相似度阈值')
    parser.add_argument('--offline_lr', type=float, default=0.001, help='Learning rate for offline model')
    parser.add_argument('--offline_optimizer', type=str, default='AdamW', help='Optimizer for offline model')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for knowledge distillation')
    # parser.add_argument('--layer_rounds', type=list, default=[30, 20, 15], help='每层训练轮数')
    parser.add_argument('--layer_rounds', type=list, default=[50, 25, 25], help='每层训练轮数')
    parser.add_argument('-dd', "--data_dir", type=str, default="../dataset/FraudDetection/")
    # 关系蒸馏参数
    parser.add_argument('--relation_kd_weight', type=float, default=0.1,help='Weight for relation knowledge distillation')
    parser.add_argument('--relation_kd_enabled', type=bool, default=True,help='Enable relation knowledge distillation')
    # PPO相关参数（服务器端）
    parser.add_argument('--ppo_epsilon', type=float, default=0.2,help='PPO clipping parameter')
    parser.add_argument('--ppo_enabled', type=bool, default=True,help='Enable PPO aggregation')

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    # if args.device == "cuda":
    #     print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    # if args.dataset == "mnist" or args.dataset == "fmnist":
    #     generate_mnist('../dataset/mnist/', args.num_clients, 10, args.niid)
    # elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
    #     generate_cifar10('../dataset/Cifar10/', args.num_clients, 10, args.niid)
    # else:
    #     generate_synthetic('../dataset/synthetic/', args.num_clients, 10, args.niid)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
