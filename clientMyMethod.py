
import copy
import os

import torch
import numpy as np
import time
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from system.flcore.clients.clientbase import Client
from system.utils.privacy import *
from system.flcore.clients.clientavg import clientAVG
from system.utils.data_utils import get_class_counts
from system.utils.data_utils import LogitHeadWrapper
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

class clientMyMethod(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 单向Teacher-Student参数
        self.momentum_tau = getattr(args, 'momentum_tau', 0.99)  # 动量系数
        self.temperature = getattr(args, 'temperature', 4.0)  # 温度参数
        self.alpha = getattr(args, 'distill_alpha', 0.7)  # 蒸馏权重

        # 初始化Teacher模型（动量更新）
        self.teacher_model = copy.deepcopy(self.model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 更保守的优化器设置
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,  # 降低学习率
            weight_decay=1e-3  # 增强正则化
        )


        self.offline_model = args.offline_model
        # self.offline_optimizer = torch.optim.SGD(self.offline_model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        self.offline_optimizer = torch.optim.Adam(
            self.offline_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # 离线模型稍弱正则化
        )
        self.classifier = copy.deepcopy(self.offline_model.classifier)
        self.other_classifier = {}


        # 改进学习率调度
        self.offline_learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.offline_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # 为在线模型也添加调度器
        self.online_learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=0.002, epochs=self.local_epochs,steps_per_epoch=len(self.load_train_data()))
        self.learning_rate_decay = True
        self.offline_learning_rate_decay = True
        self.learning_rate_decay_gamma = 0.98

        self.online_structure = {}
        self.offline_structure = {}


        # 监控相关变量
        self.anomaly_count = 0
        self.prev_loss = None
        self.detailed_logging = False
        self.local_analysis_done = False   #训练开始前调用本地数据分析

        # 自适应学习率相关参数（替代AdaptiveLRScheduler类）
        self.base_lr = args.local_learning_rate
        self.warmup_rounds = getattr(args, 'warmup_rounds', 5)
        self.decay_factor = getattr(args, 'decay_factor', 0.95)
        self.loss_history = []
        self.convergence_rate = 1.0
        self.current_round = 0
        self.last_loss = 1.0
        self.global_loss = 1.0
        self.latest_model_params = {}

        self.public_val_loader = self.load_public_validation_set()
        self.local_f1_score = 0



    def train_layered(self, layer_idx, round_idx, num_clients, alpha):
        """
        在线模型（教师）与离线模型（学生）协同训练的函数。
        两个模型共同优化，实现泛化知识与本地知识的融合。
        """

        # -------------------------------------------------------------------
        # 步骤 1: 设置模型模式、设备和优化器
        # -------------------------------------------------------------------
        trainloader = self.load_train_data()




        features = []
        labels = []
        for x, y in trainloader:
            features.append(x.view(x.size(0), -1).numpy())
            labels.append(y.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # 检查欺诈样本数量
        fraud_count = Counter(labels)[1]
        normal_count = Counter(labels)[0]

        labels_resampled_for_pos_weight = []

        # if fraud_count > 0:
        #     # 根据原始数据计算 pos_weight
        #     stable_pos_weight = torch.tensor(normal_count / fraud_count).to(self.device)
        # else:
        #     stable_pos_weight = torch.tensor(1.0).to(self.device)
        #
        # # 然后再对数据进行 SMOTE
        # # self.train_data = smote(self.train_data) # MOTE 调用
        # stable_pos_weight = torch.clamp(stable_pos_weight, min=1.0, max=5.0)
        # # 将 pos_weight 传给损失函数
        # # local_criterion = nn.BCEWithLogitsLoss(pos_weight=stable_pos_weight)
        # offline_criterion = nn.BCEWithLogitsLoss(pos_weight=stable_pos_weight)
        # online_criterion = nn.BCEWithLogitsLoss()
        # print(f"客户端 {self.id}: 本地训练，根据原始数据计算出的 pos_weight 值为: {stable_pos_weight.item():.2f}")


        # 欺诈与非欺诈不一样就用 SMOTE
        if fraud_count != normal_count:
            # print(f"客户端 {self.id}: 本地欺诈样本数量：{fraud_count}，正常样本数量：{normal_count}")

            # 只有当正负样本都存在时，才进行数据增强
            if fraud_count > 0 and normal_count > 0:
                try:
                    # 动态判断少数类，并定义过采样策略
                    if fraud_count > normal_count:
                        # 如果欺诈样本更多，则将正常样本（0）视为少数类
                        minority_class_count = normal_count
                        sampling_strategy = {0: fraud_count}
                    else:
                        # 如果正常样本更多，则将欺诈样本（1）视为少数类
                        minority_class_count = fraud_count
                        sampling_strategy = {1: normal_count}

                    # 根据少数类样本数量选择合适的过采样方法
                    if minority_class_count > 1:
                        # 如果有多个少数类样本，使用 SMOTE
                        k_neighbors = min(minority_class_count - 1, 5)
                        sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
                    else:
                        # 如果只有1个少数类样本，使用 RandomOverSampler
                        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

                    # 执行数据增强
                    features_resampled, labels_resampled = sampler.fit_resample(features, labels)
                    #打印新的样本分布，以便调试
                    # print(f"客户端 {self.id}: 增强后欺诈样本数量：{Counter(labels_resampled)[1]}，正常样本数量：{Counter(labels_resampled)[0]}")

                    labels_resampled_for_pos_weight = labels_resampled

                    # 将增强后的数据转换回 TensorDataset
                    resampled_dataset = torch.utils.data.TensorDataset(
                        torch.from_numpy(features_resampled).float(),
                        torch.from_numpy(labels_resampled).long()
                    )

                    # 创建一个新的 trainloader
                    trainloader = torch.utils.data.DataLoader(
                        dataset=resampled_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True
                    )
                except Exception as e:
                    print(f"客户端 {self.id}: 数据增强失败，跳过。错误: {e}")
            else:
                print()
                # print(f"客户端 {self.id}: 无法进行数据增强，正负样本数量之一为零。")

        # # 根据增强的样本自适应改变pos_weight
        # if len(labels_resampled_for_pos_weight) == 0:
        #     # 如果数据增强失败或没有样本，我们无法计算 pos_weight，保持默认值
        #     stable_pos_weight = torch.tensor(1.0).to(self.device)
        # else:
        #     # 使用 Counter 计算增强后的样本总数
        #     resampled_counts = Counter(labels_resampled_for_pos_weight)
        #     pos_count_total = resampled_counts[1]
        #     neg_count_total = resampled_counts[0]
        #
        #     # 避免除以零
        #     if pos_count_total > 0:
        #         stable_pos_weight = torch.tensor(neg_count_total / pos_count_total).to(self.device)
        #     else:
        #         stable_pos_weight = torch.tensor(1.0).to(self.device)
        #
        #     # 保持你原来的 pos_weight 约束
        #     stable_pos_weight = torch.clamp(stable_pos_weight, min=1.0, max=50.0)
        #     print(f"客户端 {self.id}: 本地训练，计算出的 pos_weight 值为: {stable_pos_weight.item():.2f}")
        # local_criterion = nn.BCEWithLogitsLoss(pos_weight=stable_pos_weight)


        self.model.train()  # 在线模型
        self.offline_model.train()  # 离线模型
        self.model.to(self.device)
        self.offline_model.to(self.device)


        # -------------------------------------------------------------------
        # 步骤 2: 训练循环和损失计算
        # -------------------------------------------------------------------
        epoch_losses = []

        #fednova
        num_steps = 0
        total_online_loss_sum = 0
        total_offline_loss_sum = 0

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        count = 3
        for epoch in range(max_local_epochs):
            batch_losses = []
            for step, (x, y) in enumerate(trainloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                x, y = x.to(self.device), y.to(self.device).float()
                if y.dim() > 1:
                    y = y.squeeze()
                if x.ndim > 2:
                    x = x.view(x.size(0), -1)

                # 修复点：在每个批次中动态计算 pos_weight
                # pos_count = (y == 1).sum().float()
                # neg_count = (y == 0).sum().float()
                #
                # if pos_count > 0:
                #     pos_weight = neg_count / pos_count
                #     pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)  # 保持你原来的约束
                # else:
                #     pos_weight = torch.tensor(1.0)
                # local_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))


                # 教师模型（在线模型）前向传播
                # 这里的在线模型是教师，但它也会被更新，所以不使用 torch.no_grad()
                teacher_output = self.offline_model(x)

                # 学生模型（离线模型）前向传播
                student_output = self.model(x)

                if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
                    print(f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，teacher_output 包含 NaN/Inf！")
                    # 可以在这里添加断点，或直接结束训练以定位问题
                    return
                if torch.isnan(student_output).any() or torch.isinf(student_output).any():
                    print(f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，student_output 包含 NaN/Inf！")
                    return


                # 统一输出维度
                if student_output.dim() > 1:
                    student_output = student_output.squeeze()
                if teacher_output.dim() > 1:
                    teacher_output = teacher_output.squeeze()

                local_train_online_loss = self.focal_loss(teacher_output, y, alpha=0.5, gamma=3, model_type='online')
                local_train_offline_loss = self.focal_loss(student_output, y, alpha=0.25, gamma=2, model_type='offline')

                if torch.isnan(local_train_online_loss).any() or torch.isinf(local_train_online_loss).any():
                    print(f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，online_loss 包含 NaN/Inf！")
                    return
                if torch.isnan(local_train_offline_loss).any() or torch.isinf(local_train_offline_loss).any():
                    print(f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，offline_loss 包含 NaN/Inf！")
                    return

                if round_idx != 0:
                    # 计算知识蒸馏损失
                    # # 学生（离线）向教师（在线）学习
                    # kd_loss_online_to_offlinee = self.compute_kd_loss(student_output, teacher_output, y, alpha)

                    # 额外增加一个反向的蒸馏损失，让在线模型也从离线模型的本地知识中学习
                    kd_loss_offline_to_online, ce_val, kd_val = self.compute_kd_loss(teacher_output, student_output, y, alpha)

                    # 组合总损失
                    # 在线模型总损失 = 硬标签损失 + 反向蒸馏损失
                    total_online_loss = local_train_online_loss + kd_loss_offline_to_online
                    # 离线模型总损失 = 硬标签损失 + 蒸馏损失
                    # total_offline_loss = local_train_offline_loss + kd_loss_offline_to_online
                    total_offline_loss = local_train_offline_loss

                    if step == len(trainloader) - 1 and epoch == max_local_epochs - 1:
                        print(f"客户端 {self.id} 的蒸馏损失 kd_loss: { kd_loss_offline_to_online.item():.4f}")

                # -------------------------------------------------------------------
                # 步骤 3: 损失组合和反向传播
                # -------------------------------------------------------------------
                else:
                    # 组合总损失
                    # 在线模型总损失 = 硬标签损失 + 反向蒸馏损失
                    total_online_loss = local_train_online_loss
                    # 离线模型总损失 = 硬标签损失 + 蒸馏损失
                    total_offline_loss = local_train_offline_loss



                # -------------------------------------------------------------------
                # 步骤 4: 优化两个模型
                # -------------------------------------------------------------------
                #考虑onlineloss和offlineloss权重？-----------------------------------------------------------------
                total_loss = total_online_loss + total_offline_loss
                if step == len(trainloader) - 1 and epoch == max_local_epochs - 1:
                    print(f"客户端 {self.id}: 训练轮次 {round_idx}, 本地训练第 {epoch} 轮, 在线损失: {total_online_loss.item():.4f}, 离线损失: {total_offline_loss.item():.4f}, 总损失: {total_loss.item():.4f}")

                self.optimizer.zero_grad()
                self.offline_optimizer.zero_grad()
                total_loss.backward()

                # 记录用于 FedNova 的步数和损失
                num_steps += 1
                total_online_loss_sum += total_online_loss.item()
                total_offline_loss_sum += total_offline_loss.item()

                online_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                offline_grad_norm = torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=0.1)
                if step == len(trainloader) - 1 and epoch == max_local_epochs - 1:
                    # online_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                    # offline_grad_norm = torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(),
                    #                                                    max_norm=float('inf'))

                    print(
                        f"客户端 {self.id}: 全局训练第 {round_idx}轮, 本地训练第 {epoch}轮, "
                        f"在线模型梯度范数: {online_grad_norm:.4f}, "
                        f"离线模型梯度范数: {offline_grad_norm:.4f}"
                    )

                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(
                            f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，模型 {name} 的梯度包含 NaN/Inf！")
                        return
                for name, param in self.offline_model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(
                            f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，离线模型 {name} 的梯度包含 NaN/Inf！")
                        return



                # 执行优化步骤
                self.optimizer.step()
                self.offline_optimizer.step()

                for name, param in self.model.named_parameters():
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(
                            f"警告：客户端 {self.id}，轮次 {round_idx}，本地训练第 {epoch} 轮，模型 {name} 的参数包含 NaN/Inf！")
                        return


                batch_losses.append(total_online_loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

            count = count - 1

            # -------------------------------------------------------------------
            # 步骤 5: 更新历史信息
            # -------------------------------------------------------------------
        self.online_structure[layer_idx] = copy.deepcopy(self.model).state_dict()
        self.offline_structure[layer_idx] = copy.deepcopy(self.offline_model).state_dict()

        # self.latest_model_params = {
        #     name: param for name, param in self.model.named_parameters()
        # }

        # 本地模型在公共验证集上的性能评估
        local_performance_metric = 0.0
        if self.public_val_loader is not None:
            self.model.eval()
            all_labels = []
            all_preds_binary = []
            with torch.no_grad():
                for x, y in self.public_val_loader:
                    x, y = x.to(self.device), y.to(self.device).float()
                    output = self.model(x).squeeze()
                    preds_binary = (torch.sigmoid(output) > 0.5).long()
                    all_labels.extend(y.cpu().numpy())
                    all_preds_binary.extend(preds_binary.cpu().numpy())
            try:
                local_performance_metric = f1_score(all_labels, all_preds_binary, zero_division=0)
            except Exception as e:
                print(f"客户端 {self.id}: 计算 F1-Score 失败，错误: {e}")
                local_performance_metric = 0.0
            self.local_f1_score = local_performance_metric


        # 计算平均训练损失并更新历史信息
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.training_loss = avg_loss
        global_loss = getattr(self, 'received_global_loss', self.global_loss)
        self.update_lr_history(avg_loss, global_loss, round_idx)


        # initial_params_dict = getattr(self, 'initial_params', None)
        #
        # current_params_dict = {name: param.data for name, param in self.model.named_parameters()}
        #
        # if initial_params_dict is not None:
        #     param_increment = {
        #         name: current_params_dict[name] - initial_params_dict[name]
        #         for name in current_params_dict
        #     }
        #
        #     if not self.latest_model_params:
        #         self.latest_model_params = param_increment
        #     else:
        #         for name in self.latest_model_params:
        #             self.latest_model_params[name] += param_increment[name]


        # 计算客户端模型参数相对于初始全局模型的增量
        initial_online_params = getattr(self, 'initial_params', None)

        # 仅针对可训练参数计算增量
        delta_online_params = {}
        if initial_online_params is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # 检查参数是否需要梯度
                    # 确保 initial_online_params 中包含该键，以防意外
                    if name in initial_online_params:
                        # 计算当前参数与初始全局参数的差值
                        delta_online_params[name] = param.data - initial_online_params[name]
        else:
            # 在 round 0，如果无法获取初始全局模型，则使用本地模型的所有可训练参数
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    delta_online_params[name] = param.data

        self.latest_model_params = {
            'model_params': self.model.state_dict(),
            'num_samples': self.train_samples
        }

    def focal_loss(self,inputs, targets, alpha=0.25, gamma=2, reduction='mean', model_type='online'):
        """
        Focal Loss for binary classification.
        :param inputs: Model outputs (logits)
        :param targets: True labels (0 or 1)
        :param alpha: Class balancing factor for positive class
        :param gamma: Focusing parameter to down-weight easy examples
        :param reduction: 'mean' or 'sum', specifies the reduction to apply
        :param model_type: 'online' or 'offline' to distinguish between local and global models
        :return: Computed focal loss value
        """
        # Binary Cross-Entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt (probability predicted by model)
        pt = torch.exp(-bce_loss)

        # Apply Focal Loss
        focal_loss_value = alpha * (1 - pt) ** gamma * bce_loss

        # Apply reduction (mean or sum)
        if reduction == 'mean':
            return focal_loss_value.mean()
        elif reduction == 'sum':
            return focal_loss_value.sum()
        else:
            return focal_loss_value


    def set_parameters_with_classifier(self, global_structure, classifier_list, layer):


        local_online_state_dict = self.model.state_dict()

        # 遍历服务器传来的全局参数字典
        for key, new_tensor in global_structure.items():
            # 检查是否为BatchNorm层的参数
            if any(substr in key for substr in ["bn", "running_mean", "running_var", "num_batches_tracked"]):
                # 如果是BN层参数，则跳过不更新
                continue

            # 对于所有其他参数，进行更新
            if key in local_online_state_dict and local_online_state_dict[key].shape == new_tensor.shape:
                local_online_state_dict[key].copy_(new_tensor)

        # 将修改后的本地状态字典加载回在线模型
        self.model.load_state_dict(local_online_state_dict)
        self.model.to(self.device)

        # self.online_structure[current_layer] = copy.deepcopy(self.model)

        # temp_offline_model = copy.deepcopy(self.offline_model)
        # temp_offline_model.to(self.device)  # 确保复制的模型在正确设备上
        # self.offline_structure[current_layer] = temp_offline_model.state_dict()
        # self.offline_structure[current_layer] = copy.deepcopy(self.offline_model)

        if classifier_list != {}:
            self.other_classifier = {}

            for id, classifier in classifier_list.items():
                # if id != self.id:
                self.other_classifier[id] = copy.deepcopy(classifier)

        self.initial_params = {name: param.data.clone().detach().to(self.device) for name, param in self.model.named_parameters()}
        self.initial_offline_params = {name: param.data.clone().detach().to(self.device)for name, param in self.offline_model.named_parameters()}





#######################################################################################################################################################




    def log_gradient_norms(self):
        """只在异常时记录梯度范数"""
        if not self.detailed_logging:
            return None

        online_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
        offline_norm = sum(p.grad.norm().item() for p in self.offline_model.parameters() if p.grad is not None)

        if online_norm > 5.0 or offline_norm > 5.0:
            print(f"    [CLIENT {self.id}] GRAD WARNING: Online={online_norm:.3f}, Offline={offline_norm:.3f}")

        return online_norm, offline_norm

    def log_extreme_activations(self, model_output):
        """只在激活值异常时记录"""
        if not self.detailed_logging:
            return

        if model_output.abs().max() > 10:
            print(f"    [CLIENT {self.id}] ACTIVATION WARNING: Max={model_output.abs().max().item():.3f}")




    def get_client_fraud_level(self):
        """
        在客户端本地计算欺诈比例并映射为离散等级（整数）。
        服务器只接收等级（0,1,2），而不会看到具体数量或比例。

        等级示例（可按需要调整阈值）：
        level 0: 低欺诈 (ratio <= 0.5%)
        level 1: 中等欺诈 (0.5% < ratio <= 1.5%)
        level 2: 高欺诈 (ratio > 1.5%)
        """
        try:
            # 尽量从已有的本地标签数据获取，兼容不同数据结构
            if hasattr(self, 'y_train'):
                y = self.y_train
            else:
                # 退而求其次，从样本集中载入一遍（代价低，且本地）
                loader = self.load_train_data()
                all_labels = []
                for _, batch_y in loader:
                    # 支持 list 或 tensor
                    if type(batch_y) == type([]):
                        batch_y = batch_y[0]
                    all_labels.append(batch_y.cpu().numpy())
                if len(all_labels) == 0:
                    return 0
                y = np.concatenate(all_labels)
            # 计算比例
            if isinstance(y, torch.Tensor):
                num_fraud = int((y == 1).sum().item())
                num_total = int(len(y))
            else:
                num_fraud = int(np.sum(np.array(y) == 1))
                num_total = int(len(y))
            ratio = (num_fraud / num_total) if num_total > 0 else 0.0
        except Exception:
            # 出于鲁棒性考虑，若任何异常则返回中间等级
            return 1

        # 阈值按你的数据集特征调整
        if ratio <= 0.005:
            return 0
        elif ratio <= 0.015:
            return 1
        else:
            return 2

#----------------------------------------------------------------------------------------------------------------
    # def update_teacher_momentum(self):
    #     """动量更新Teacher模型"""
    #     self.teacher_model.to(self.device)
    #     self.model.to(self.device)
    #     with torch.no_grad():
    #         for teacher_param, student_param in zip(
    #                 self.teacher_model.parameters(),
    #                 self.model.parameters()
    #         ):
    #             teacher_param.data = (
    #                     self.momentum_tau * teacher_param.data +
    #                     (1 - self.momentum_tau) * student_param.data
    #             )

    def compute_kd_loss(self,teacher_logits, student_logits, labels, alpha):
        """
           计算带温度系数的蒸馏损失。
           :param teacher_logits: 教师模型的 logits（未经 sigmoid 处理）
           :param student_logits: 学生模型的 logits（未经 sigmoid 处理）
           :param labels: 真实标签（用于 CE 损失）
           :param alpha: 权重，控制 CE 损失和 KD 损失的比例
           """
        T = self.temperature  # 温度系数
        # 保证 logits 形状为 (B,1)
        if teacher_logits.dim() == 1:
            teacher_logits = teacher_logits.unsqueeze(1)
        if student_logits.dim() == 1:
            student_logits = student_logits.unsqueeze(1)

        # 学生模型的监督损失（交叉熵）
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            labels = labels.float().unsqueeze(1)
            ce_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

        # KD 损失：教师模型的软目标与学生模型的软目标之间的 KL 散度
        t2 = self.to_2class_logits(teacher_logits) / T
        s2 = self.to_2class_logits(student_logits) / T

        # 计算教师的 softmax 输出和学生的 log-softmax
        p_teacher = F.softmax(t2, dim=1)
        log_p_student = F.log_softmax(s2, dim=1)

        # KL 散度
        kd_loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T * T)

        # 加权比例
        alpha_val = alpha
        beta_val = 1.0 - alpha_val

        # 总损失（加权交叉熵损失和 KD 损失）
        total_loss = alpha_val * ce_loss + beta_val * kd_loss
        return total_loss, ce_loss.detach() if labels is not None else None, kd_loss.detach()

    def to_2class_logits(self,x):
        return torch.cat([x, torch.zeros_like(x)], dim=1)  # shape (B,2)


    # def train_layer_specific(self, layer_idx, round_idx, num_clients, alpha):#已经废弃！！！！！！！！！！！！！！！！！
    #     """单向teacher-student架构的层级训练"""
    #     trainloader = self.load_train_data()
    #     # self.model.train()
    #     # self.teacher_model.eval()
    #     #
    #     # self.teacher_model.to(self.device)
    #     # self.model.to(self.device)
    #
    #     # 确保正在训练正确的层级
    #     if layer_idx not in self.online_structure:
    #         print(f"警告: 客户端 {self.id} 在 {layer_idx} 层找不到模型")
    #         return
    #
    #     # 获取当前层的学生和教师模型模块
    #     student_model_layer = self.online_structure[layer_idx]
    #     teacher_model_layer = self.offline_structure[layer_idx]
    #
    #     student_model_layer.train()
    #     teacher_model_layer.eval()
    #     self.model.train()
    #     self.offline_model.eval()
    #     self.model.to(self.device)
    #     self.offline_model.to(self.device)
    #     student_model_layer.to(self.device)
    #     teacher_model_layer.to(self.device)
    #
    #     # === 设置自适应学习率 ===
    #     optimizer = torch.optim.Adam(student_model_layer.parameters(), lr=self.get_adaptive_lr())
    #
    #     trainloader = self.load_train_data()
    #     epoch_losses = []
    #
    #     epoch_losses = []
    #
    #     # 本地训练轮次
    #     for epoch in range(self.local_epochs):
    #         batch_losses = []
    #
    #         for step, (x, y) in enumerate(trainloader):
    #             if self.train_slow:
    #                 time.sleep(0.1 * np.abs(np.random.rand()))
    #
    #             x, y = x.to(self.device), y.to(self.device).float()
    #             if y.dim() > 1:
    #                 y = y.squeeze()
    #             if x.ndim > 2:
    #                 x = x.view(x.size(0), -1)
    #
    #             with torch.no_grad():
    #                 # 教师模型前向传播
    #                 teacher_features = self.offline_model.produce_feature(x)
    #                 teacher_output = self.offline_model.classifier(teacher_features)
    #
    #             # 学生模型前向传播
    #             # 如果训练的是分类器，需要先用特征提取器获取特征
    #             if layer_idx == 1:
    #                 with torch.no_grad():
    #                     student_features = self.model.produce_feature(x)
    #                 student_output = student_model_layer(student_features)
    #             # 如果训练的是特征提取器，直接前向传播
    #             else:
    #                 student_features = student_model_layer.produce_feature(x)
    #                 student_output = self.model.classifier(student_features)
    #
    #             if student_output.dim() > 1:
    #                 student_output = student_output.squeeze()
    #             if teacher_output.dim() > 1:
    #                 teacher_output = teacher_output.squeeze()
    #
    #             # 计算知识蒸馏损失
    #             loss = self.compute_kd_loss(student_output, teacher_output, y, alpha)
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             # 梯度裁剪防止爆炸
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    #             optimizer.step()
    #             batch_losses.append(loss.item())
    #
    #         epoch_loss = sum(batch_losses) / len(batch_losses)
    #         epoch_losses.append(epoch_loss)
    #
    #         # 每个epoch结束后更新teacher
    #         self.update_teacher_momentum()
    #
    #     # 计算平均训练损失并更新历史信息
    #     avg_loss = sum(epoch_losses) / len(epoch_losses)
    #     self.training_loss = avg_loss  # 保存训练损失供服务器使用
    #
    #     # 注意：global_loss需要从服务器获取，这里暂时使用1.0
    #     global_loss = getattr(self, 'received_global_loss', self.global_loss)
    #     self.update_lr_history(avg_loss, global_loss, round_idx)
    #
    #     print(f"[CLIENT {self.id}] 第{layer_idx}层训练完成, 平均损失: {avg_loss:.4f}")

    # def get_adaptive_lr(self):
    #     """基于性能自适应调整学习率"""
    #     # Warmup阶段
    #     if self.current_round < self.warmup_rounds:
    #         return self.base_lr * (self.current_round + 1) / self.warmup_rounds
    #
    #     # 基于损失变化率调整
    #     if len(self.loss_history) >= 2:
    #         recent_change = abs(self.loss_history[-1] - self.loss_history[-2])
    #         if recent_change < 0.01:  # 收敛缓慢
    #             self.convergence_rate *= 1.1
    #         elif recent_change > 0.5:  # 震荡过大
    #             self.convergence_rate *= 0.8
    #
    #     # 基于全局-本地性能差异
    #     performance_gap = abs(self.last_loss - self.global_loss) / (self.global_loss + 1e-8)
    #     gap_factor = 1.0 + 0.2 * performance_gap
    #
    #     adaptive_lr = (self.base_lr *
    #                    self.convergence_rate *
    #                    gap_factor *
    #                    (self.decay_factor ** (self.current_round - self.warmup_rounds)))
    #
    #     # 限制学习率范围
    #     return max(1e-5, min(0.01, adaptive_lr))

    def update_lr_history(self, loss_value, global_loss_value, round_idx):
        """更新学习率调整所需的历史信息"""
        self.loss_history.append(loss_value)
        self.last_loss = loss_value
        self.global_loss = global_loss_value
        self.current_round = round_idx

        # 只保留最近10次历史
        if len(self.loss_history) > 10:
            self.loss_history = self.loss_history[-10:]

    def get_noisy_pos_ratio(self):
        """
        计算并返回带有差分隐私噪声的正样本比例。
        """
        pos_count = 0
        total_count = 0
        trainloader = self.load_train_data()
        # 遍历整个数据集来计算正负样本数
        for _, labels in trainloader:
            pos_count += (labels == 1).sum().item()
            total_count += len(labels)

        true_pos_ratio = pos_count / total_count if total_count > 0 else 0.0

        # 定义差分隐私参数 epsilon
        privacy_epsilon = 0.5

        # 计算拉普拉斯噪声的尺度参数 b
        laplace_b = 1.0 / privacy_epsilon

        # 生成拉普拉斯噪声并添加到真实比例中
        noise = np.random.laplace(loc=0.0, scale=laplace_b)
        noisy_pos_ratio = true_pos_ratio + noise

        # 将近似值限制在 [0, 1] 范围内
        return max(0.0, min(1.0, noisy_pos_ratio))

    def load_public_validation_set(self):
        """加载公共验证集"""
        public_val_path = "C:\\Users\\Study\\PycharmProjects\\swu-fl\\dataset\\FraudDetection\\public_val_set.npz"
        if os.path.exists(public_val_path):
            data = np.load(public_val_path, allow_pickle=True)
            X_val, y_val = data['x'], data['y']
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                batch_size=self.batch_size,
                shuffle=False
            )
        print("警告: 客户端无法找到公共验证集，基于性能的聚合将退化为平均聚合。")
        return None

    def get_sample_weight(self, labels, student_output):
        """
        根据 AUPRC 计算样本权重。
        :param labels: 真实标签
        :param student_output: 模型输出（通常是概率值）
        :return: 样本权重（基于 AUPRC）
        """
        # 计算 AUPRC（平均精度得分）
        auprc = average_precision_score(labels.cpu().numpy(), student_output.cpu().detach().numpy())

        # 基于 AUPRC 计算样本的权重（可以调整公式）
        weight = auprc  # 你可以根据需求进一步调整公式

        return weight
# ----------------------------------------------------------------------------------------------------------------