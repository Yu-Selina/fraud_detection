
import copy
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

class clientMyMethod(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 在线模型：标准BCE
        self.loss = nn.BCEWithLogitsLoss()

        # 离线模型：加权BCE，给正样本更高权重
        def get_weighted_bce():
            def weighted_bce_loss(inputs, targets):
                pos_count = (targets == 1).sum()
                neg_count = (targets == 0).sum()

                if pos_count > 0:
                    pos_weight = neg_count.float() / pos_count.float()
                    pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)  # 限制权重范围
                else:
                    pos_weight = torch.tensor(1.0)

                return F.binary_cross_entropy_with_logits(
                    inputs, targets,
                    pos_weight=pos_weight,
                    reduction='mean'
                )

            return weighted_bce_loss

        self.offline_loss = get_weighted_bce()


        # self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.offline_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        # —— INSERT —— 为 online/offline 各建一份 EMA 教师（不参与梯度）
        self.online_teacher = copy.deepcopy(self.model)
        self.offline_teacher = copy.deepcopy(self.offline_model)
        for p in self.online_teacher.parameters():
            p.requires_grad_(False)
        for p in self.offline_teacher.parameters():
            p.requires_grad_(False)

        # —— INSERT —— EMA 衰减与蒸馏缓启（可从 args 读取，给默认值）
        self.ema_tau = getattr(args, 'ema_tau', 0.99)
        self.kd_warmup_rounds = getattr(args, 'kd_warmup_rounds', 10)

        self.kd_temperature = getattr(args, 'kd_temperature', 4.0)
        self.kd_alpha = getattr(args, 'kd_alpha', 0.7)
        self.kd_beta = getattr(args, 'kd_beta', 0.3)

        # 改进学习率调度
        self.offline_learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.offline_optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        # 为在线模型也添加调度器
        self.online_learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=0.002, epochs=self.local_epochs,steps_per_epoch=len(self.load_train_data()))
        self.learning_rate_decay = True
        self.offline_learning_rate_decay = True
        self.learning_rate_decay_gamma = 0.98

        self.temperature = 2.0   #温度系数
        self.alpha_online = 0.7  # 在线模型的硬标签权重
        self.alpha_offline = 0.7  # 离线模型的硬标签权重

        self.online_structure = {}  # 存每个层级的在线模型
        self.offline_structure = {} # 存每个层级的离线模型

        self.relation_kd_weight = getattr(args, 'relation_kd_weight', 0.1)
        self.relation_kd_enabled = getattr(args, 'relation_kd_enabled', True)

        # 监控相关变量
        self.anomaly_count = 0
        self.prev_loss = None
        self.detailed_logging = False
        self.local_analysis_done = False   #训练开始前调用本地数据分析





    def train(self,global_rounds,num_clients):

        sum_local_online_loss = 0.0
        sum_local_offline_loss = 0.0
        sum_classifier_online_loss = 0.0
        sum_classifier_offline_loss = 0.0
        sum_total_online_loss = 0.0
        sum_total_offline_loss = 0.0
        num_batches = 0

        # 添加L2正则化
        l2_lambda = 1e-4  # L2正则化系数

        # KL散度损失函数，用于度量两个分布的差异
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        trainloader = self.load_train_data()  # 加载训练数据
        self.model.to(self.device)
        self.offline_model.to(self.device)
        self.model.train()  #在线模型
        self.offline_model.train()  #离线模型
        if not self.local_analysis_done:
            self.analyze_data_locally(trainloader)
            self.local_analysis_done = True

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        total_gradient_norm = 0.0  # 总梯度范数
        total_batches = 0  # 总批次数

        if global_rounds != 0:  #双向知识蒸馏过程

            # —— EDIT —— 本轮蒸馏缓启系数（供 compute_distillation_loss 使用）
            self._current_kd_ramp = self._kd_ramp(global_rounds)

            # last_online_model = copy.deepcopy(self.model)
            # last_online_model.eval()
            # last_offline_model = copy.deepcopy(self.offline_model)
            # last_offline_model.eval()

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(
                        trainloader):  # 也可以是for  x,y in trainloader： 区别只是原文中的可以获取每个训练批次的数据和索引，而注释里这个简化版只能直接获取每个批次的数据而不能取得对应的索引
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))


                    online_output = self.model(x)
                    offline_output = self.offline_model(x)

                    # with torch.no_grad():
                    #     last_online_output = last_online_model(x)
                    #     last_offline_output = last_offline_model(x)


                    # —— EDIT —— 教师来自 EMA，不参与梯度
                    with torch.no_grad():
                        last_online_output = self.online_teacher(x)
                        last_offline_output = self.offline_teacher(x)

                    self.optimizer.zero_grad()

                    dist_online_loss, dist_offline_loss = self.compute_distillation_loss(online_output,offline_output,last_online_output,last_offline_output,y)

                    # —— EDIT —— 交替更新：先在线
                    mu = getattr(self, 'fedprox_mu', 0.0)

                    if mu > 0:
                        global_snapshot = getattr(self, 'initial_params', None)
                        offline_snapshot = getattr(self, 'initial_offline_params', None)

                        if global_snapshot is not None:
                            prox_term = self.apply_fedprox_proximal(self.model, global_snapshot, mu)
                            dist_online_loss = dist_online_loss + prox_term

                        if offline_snapshot is not None:
                            prox_term_off = self.apply_fedprox_proximal(self.offline_model, offline_snapshot, mu)
                            dist_offline_loss = dist_offline_loss + prox_term_off
                    self.optimizer.zero_grad()
                    dist_online_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    # 再离线
                    self.offline_optimizer.zero_grad()
                    dist_offline_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=0.5)
                    self.offline_optimizer.step()

                    # —— EDIT —— 每步后更新 EMA 教师
                    self._ema_update(self.online_teacher, self.model)
                    self._ema_update(self.offline_teacher, self.offline_model)



        for epoch in range(max_local_epochs):   #本地训练阶段
            for i, (x, y) in enumerate(
                    trainloader):  # 也可以是for  x,y in trainloader： 区别只是原文中的可以获取每个训练批次的数据和索引，而注释里这个简化版只能直接获取每个批次的数据而不能取得对应的索引
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                online_feature = self.model.produce_feature(x)   #提取特征并给出输出
                online_output = self.model.classifier(online_feature).squeeze(1)
                offline_feature = self.offline_model.produce_feature(x)
                offline_output = self.offline_model.classifier(offline_feature).squeeze(1)

                # y = y.float().view(-1, 1).to(self.device)
                y = y.float().view(-1).to(self.device)
                # self.optimizer.zero_grad()
                # local_train_online_loss = self.balanced_softmax_loss(y,online_output,self.sample_per_class)
                local_train_online_loss= self.loss(online_output,y)  #在线模型本地训练损失
                # local_train_offline_loss = self.balanced_softmax_loss(y,offline_output,self.sample_per_class)
                local_train_offline_loss = self.offline_loss(offline_output,y)   #离线模型本地训练损失

                # 关系蒸馏 - 在反向传播前添加
                if global_rounds > 10 and self.relation_kd_enabled:  # 原版是 global_rounds > 0
                    offline_feature_detached = offline_feature.detach()
                    online_feature_detached = online_feature.detach()

                    relation_kd_loss_online = self.compute_relation_distillation(online_feature,
                                                                                 offline_feature_detached)
                    relation_kd_loss_offline = self.compute_relation_distillation(offline_feature,
                                                                                  online_feature_detached)

                    # 降低知识蒸馏权重（原版self.relation_kd_weight，现在乘以0.5）
                    kd_weight = self.relation_kd_weight * 0.5
                    total_online_loss = local_train_online_loss + kd_weight * relation_kd_loss_online
                    total_offline_loss = local_train_offline_loss + kd_weight * relation_kd_loss_offline
                else:
                    total_online_loss = local_train_online_loss
                    total_offline_loss = local_train_offline_loss

                mu = getattr(self, 'fedprox_mu', 0.0)
                if mu > 0:
                    global_snapshot = getattr(self, 'initial_params', None)
                    offline_snapshot = getattr(self, 'initial_offline_params', None)

                    if global_snapshot is not None:
                        prox_online = self.apply_fedprox_proximal(self.model, global_snapshot, mu)
                        total_online_loss = total_online_loss + prox_online

                    if offline_snapshot is not None:
                        prox_offline = self.apply_fedprox_proximal(self.offline_model, offline_snapshot, mu)
                        total_offline_loss = total_offline_loss + prox_offline

                self.optimizer.zero_grad()
                total_online_loss.backward()
                self.optimizer.step()

                self.offline_optimizer.zero_grad()
                total_offline_loss.backward()
                self.offline_optimizer.step()

                self._ema_update(self.online_teacher, self.model)
                self._ema_update(self.offline_teacher, self.offline_model)

                # 累计总损失
                sum_total_online_loss += total_online_loss.item()
                sum_total_offline_loss += total_offline_loss.item()
                # print(f"Final - total_online_loss: {total_online_loss.item()}")
                # print(f"Final - total_offline_loss: {total_offline_loss.item()}")
                # 批次计数
                num_batches += 1


                # 计算当前批次的梯度2范数并累加到总梯度范数中
                batch_norm = 0.0
                for name, p in self.model.named_parameters():
                    if name != 'scaling' and p.grad is not None:  # 明确排除 scaling 参数
                        param_norm = p.grad.data.norm(2)
                        batch_norm += param_norm.item() ** 2
                batch_norm = batch_norm ** 0.5
                total_gradient_norm += batch_norm
                total_batches += 1
            if epoch == max_local_epochs - 1:  # 只在最后一轮本地训练记录
                with torch.no_grad():
                    # 快速测试一个batch
                    test_batch_x, test_batch_y = next(iter(trainloader))

                    # 关键修复：确保测试数据在GPU上
                    if type(test_batch_x) == type([]):
                        test_batch_x[0] = test_batch_x[0].to(self.device)
                    else:
                        test_batch_x = test_batch_x.to(self.device)
                    test_batch_y = test_batch_y.to(self.device)

                    # 现在可以安全调用模型
                    online_output = self.model(test_batch_x)
                    offline_output = self.offline_structure['全局'](
                        test_batch_x) if '全局' in self.offline_structure else self.offline_model(test_batch_x)

                    online_acc = ((torch.sigmoid(online_output) > 0.5) == test_batch_y).float().mean()
                    offline_acc = ((torch.sigmoid(offline_output) > 0.3) == test_batch_y).float().mean()

                    # 检测异常
                    anomaly = self.check_anomaly(sum_total_online_loss, sum_total_offline_loss, online_acc, offline_acc)

                    # 记录梯度信息（仅异常时）
                    self.log_gradient_norms()

                    # 记录激活值（仅异常时）
                    self.log_extreme_activations(online_output)
                    self.log_extreme_activations(offline_output)

                    # 简化输出：每个客户端训练完只打印一行
                    if anomaly or self.id == 0:  # 异常时或第一个客户端时打印
                        print(
                            f"[CLIENT {self.id}] Loss: Online={sum_total_online_loss:.3f} Offline={sum_total_offline_loss:.3f} | Acc: Online={online_acc:.3f} Offline={offline_acc:.3f} {'⚠️' if anomaly else ''}")

        # 计算平均训练损失用于相似度计算
        if num_batches > 0:
            self.current_train_loss = sum_total_online_loss / num_batches
        else:
            self.current_train_loss = 0.0

        # 记录模型更新（如果有初始参数记录）
        if hasattr(self, 'initial_params'):
            model_update = {}
            for name, param in self.model.named_parameters():
                if name in self.initial_params:
                    model_update[name] = param.data - self.initial_params[name]
            # 这个更新将在服务器端的receive函数中被收集

        self.grad_norm.append(total_gradient_norm / total_batches)

        if self.learning_rate_decay:
            self.online_learning_rate_scheduler.step()
        if self.offline_learning_rate_decay:
            self.offline_learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def compute_distillation_loss(self,
                                  current_online_output,  # 本轮在线模型输出 (batch,1)
                                  current_offline_output,  # 本轮离线模型输出 (batch,1)
                                  last_online_output,  # 上一轮在线模型输出 (batch,1)
                                  last_offline_output,  # 上一轮离线模型输出 (batch,1)
                                  labels):
        """跨轮次的双向知识蒸馏（二分类 + BCEWithLogitsLoss版本）"""
        labels = labels.float().view(-1)

        # 硬标签
        online_hard = self.loss(current_online_output, labels)
        offline_hard = self.offline_loss(current_offline_output, labels)

        # 软标签：teacher 概率（logit/T 后过 sigmoid）
        T = self.temperature
        with torch.no_grad():
            t_offline_prob = torch.sigmoid(last_offline_output / T)
            t_online_prob = torch.sigmoid(last_online_output / T)

        online_soft = F.binary_cross_entropy_with_logits(
            current_online_output / T, t_offline_prob, reduction='mean'
        ) * (T ** 2)

        offline_soft = F.binary_cross_entropy_with_logits(
            current_offline_output / T, t_online_prob, reduction='mean'
        ) * (T ** 2)

        ao = getattr(self, 'alpha_online', 0.7)
        af = getattr(self, 'alpha_offline', 0.7)

        # —— EDIT —— 蒸馏项乘以缓启系数
        ramp = getattr(self, '_current_kd_ramp', 1.0)
        total_online_loss = ao * online_hard + ramp * (1 - ao) * online_soft
        total_offline_loss = af * offline_hard + ramp * (1 - af) * offline_soft
        return total_online_loss, total_offline_loss
        # # 1. 硬标签监督（BCEWithLogitsLoss）
        # # labels = labels.float().view(-1, 1)
        # labels = labels.float().view(-1)
        # online_hard_loss = self.loss(current_online_output, labels)
        # offline_hard_loss = self.offline_loss(current_offline_output, labels)
        #
        # # 2. 软标签蒸馏（基于 sigmoid 概率）
        # T = self.temperature
        #
        # # 学生与教师的概率
        # student_online_prob = torch.sigmoid(current_online_output / T)
        # teacher_offline_prob = torch.sigmoid(last_offline_output.detach() / T)
        #
        # student_offline_prob = torch.sigmoid(current_offline_output / T)
        # teacher_online_prob = torch.sigmoid(last_online_output.detach() / T)
        #
        # # KL 散度：这里我们用 log(p_student) vs p_teacher
        # online_soft_loss = F.kl_div(
        #     torch.log(student_online_prob + 1e-8),
        #     teacher_offline_prob,
        #     reduction="batchmean"
        # ) * (T ** 2)
        #
        # offline_soft_loss = F.kl_div(
        #     torch.log(student_offline_prob + 1e-8),
        #     teacher_online_prob,
        #     reduction="batchmean"
        # ) * (T ** 2)
        #
        # # 3. 总损失
        # total_online_loss = 0.5 * online_hard_loss + 0.5 * online_soft_loss
        # total_offline_loss = 0.5 * offline_hard_loss + 0.5 * offline_soft_loss


    def kl_loss_with_temperature(self, student_output, teacher_output, temperature):
        """适配二分类单 logit 输出的 KL 散度损失"""

        # 概率化（sigmoid + 温度缩放）
        student_prob = torch.sigmoid(student_output / temperature)
        teacher_prob = torch.sigmoid(teacher_output / temperature)

        # 避免 log(0)
        student_log_prob = torch.log(student_prob + 1e-8)

        # KL(P_teacher || P_student)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

        return loss * (temperature ** 2)



    def set_parameters_with_classifier(self, model,classifier_list,global_structure):

        layers = [list(global_structure.keys())[-1]]
        # for layer in layers:
        current_layer = layers[-1]
        for group_id, group in global_structure[current_layer].items():
            if self.id in group[0]:
                # self.online_structure[layer] = copy.deepcopy(group[1])
                group_state = group[1].state_dict()
                local_state = self.model.state_dict()
                for key, new_tensor in group_state.items():
                    # 跳过 Online 模型的 BatchNorm 参数
                    if any(substr in key for substr in ["bn", "running_mean", "running_var", "num_batches_tracked"]):
                        continue
                    # 其他参数照常更新
                    if key in local_state and local_state[key].shape == new_tensor.shape:
                        local_state[key].copy_(new_tensor)# 直接把新模型的值 copy_ 到旧模型对应项
                # 【插入点 A】：包裹 online 子模型
                temp_model = copy.deepcopy(self.model)
                temp_model.to(self.device)
                self.online_structure[current_layer] = LogitHeadWrapper(temp_model, self.num_classes).to(self.device)

                # self.online_structure[current_layer] = copy.deepcopy(self.model)
        # 【插入点 B】：包裹 offline 子模型
        temp_offline_model = copy.deepcopy(self.offline_model)
        temp_offline_model.to(self.device)  # 确保复制的模型在正确设备上
        self.offline_structure[current_layer] = LogitHeadWrapper(temp_offline_model, self.num_classes).to(self.device)
        # self.offline_structure[current_layer] = copy.deepcopy(self.offline_model)

        if classifier_list != {}:
            self.other_classifier = {}

            for id, classifier in classifier_list.items():
                # if id != self.id:
                self.other_classifier[id] = copy.deepcopy(classifier)

        self.initial_params = {name: param.data.clone().detach().to(self.device) for name, param in self.model.named_parameters()}
        self.initial_offline_params = {name: param.data.clone().detach().to(self.device)for name, param in self.offline_model.named_parameters()}
                    # if classifier_list != {}:
                    #     self.other_classifier = {}
                    #
                    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                    #         for key in model.state_dict().keys():
                    #             if 'bn' not in key:
                    #                 old_param.data = new_param.data.clone()
                    #     for id,classifier in classifier_list.items():
                    #         # if id != self.id:
                    #         self.other_classifier[id] = copy.deepcopy(classifier)
                    #
                    # else:
                    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                    #         for key in model.state_dict().keys():
                    #             if 'bn' not in key:
                    #                 old_param.data = new_param.data.clone()


    def compute_consistency_loss(self, online_features, offline_features):
        """添加特征一致性损失"""
        return F.mse_loss(online_features, offline_features.detach())

    def test_metrics(self):
        testloaderfull = self.load_test_data()  # 加载测试数据
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.to(self.device)
        self.offline_model.to(self.device)
        for layer, model in self.online_structure.items():
            model.eval()
        for layer, model in self.offline_structure.items():
            model.eval()
        self.model.eval()
        self.offline_model.eval()

        online_test_correct_num = 0  # 测试准确的数量
        offline_test_correct_num = 0
        total_test_correct_num = 0
        test_num = 0  # 样本数
        y_prob = []  # 预测结果列表
        y_true = []  # 真实标签列表


        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)  # 如果x是列表形式，那么就输出列表的第一个输入源的数据
                else:
                    x = x.to(self.device)  # 将输入数据x移动到设备上（默认是GPU）
                y = y.to(self.device)  # 将标签y移动到设备上（默认是GPU）


                total_online_output = None
                online_model_count = 0

                for layer, online_model in self.online_structure.items():
                    online_output = online_model(x).squeeze()

                    if online_output.dim() == 2:
                        if online_output.shape[1] == 2:
                            online_output = online_output[:, 1] - online_output[:, 0]
                        elif online_output.shape[1] == 1:
                            online_output = online_output[:, 0]
                        else:
                            raise ValueError(
                                f"Unexpected output shape {online_output.shape} for binary classification.")

                    # 确保是1D tensor
                    online_output = online_output.view(-1).to(self.device)

                    # 安全的累积方式
                    if total_online_output is None:
                        total_online_output = online_output.clone()
                    else:
                        total_online_output += online_output

                    online_model_count += 1

                # 计算平均值
                if online_model_count > 0:
                    total_online_output = total_online_output / float(online_model_count)

                total_offline_output = None
                offline_model_count = 0

                for layer, offline_model in self.offline_structure.items():
                    offline_output = offline_model(x).squeeze()

                    if offline_output.dim() == 2:
                        if offline_output.shape[1] == 2:
                            offline_output = offline_output[:, 1] - offline_output[:, 0]
                        elif offline_output.shape[1] == 1:
                            offline_output = offline_output[:, 0]
                        else:
                            raise ValueError(
                                f"Unexpected output shape {offline_output.shape} for binary classification.")

                    # 确保是1D tensor
                    offline_output = offline_output.view(-1).to(self.device)

                    # 安全的累积方式
                    if total_offline_output is None:
                        total_offline_output = offline_output.clone()
                    else:
                        total_offline_output += offline_output

                    offline_model_count += 1

                # 计算平均值
                if offline_model_count > 0:
                    total_offline_output = total_offline_output / float(offline_model_count)


                # 确保y的维度正确
                y_targets = y.float().view(-1)
                # 检查并修正batch size不匹配问题
                if total_online_output.size(0) != y_targets.size(0):
                    # 如果batch size不匹配，截取或填充到匹配的大小
                    min_batch_size = min(total_online_output.size(0), y_targets.size(0))
                    total_online_output = total_online_output[:min_batch_size]
                    y_targets = y_targets[:min_batch_size]
                online_loss = self.loss(total_online_output, y_targets)

                # 同样的检查和修正
                if total_offline_output.size(0) != y_targets.size(0):
                    min_batch_size = min(total_offline_output.size(0), y_targets.size(0))
                    total_offline_output = total_offline_output[:min_batch_size]
                    # y_targets已经在上面处理过了，这里需要重新获取
                    y_targets_offline = y.float().view(-1)[:min_batch_size]
                else:
                    y_targets_offline = y_targets
                offline_loss = self.offline_loss(total_offline_output, y_targets_offline)

            #根据loss动态分配online和offline输出的权重占比
##################################################################################################################################################################
                # # 这里的online_weight、offline_weight是用于监测在测试集上的表现，self.online_weight等带self的则是在训练集上的表现
                # online_weight, offline_weight = self.compute_dynamic_weights(online_loss, offline_loss)
                # total_output = online_weight * online_output.detach() + offline_weight * offline_output.detach()
                # if self.id == 1:
                #     print("Client", self.id,
                #           " Test metrics - Online_loss:", online_loss.item(),
                #           " Offline_loss:", offline_loss.item(),
                #           " Online_weight:", online_weight.item(),
                #           " Offline_weight:", offline_weight.item())
##################################################################################################################################################################


            #后续考虑根据层级递增逐渐增加offline输出占比，降低online输出占比
##################################################################################################################################################################
                # total_output = (total_online_output + total_offline_output) / 2
                #
                # preds = torch.argmax(total_output, dim=1)  # [batch] 预测类别，原代码是用 argmax，这是不需要的，因为只有两个类别（非欺诈/欺诈）
                total_output = (total_online_output + total_offline_output) / 2
                # —— EDIT —— 分开统计 online/offline，再统计 fused
                online_probs = torch.sigmoid(total_online_output)
                offline_probs = torch.sigmoid(total_offline_output)
                fused_probs = torch.sigmoid(total_output)

                # 为离线模型使用更低的阈值（更容易预测为正类）
                offline_threshold = 0.3  # 默认是0.5
                online_preds = (online_probs > 0.5).long()
                offline_preds = (offline_probs > offline_threshold).long()  # 降低阈值
                fused_preds = (fused_probs > 0.4).long()  # 融合模型也稍微降低阈值

                online_test_correct_num += (online_preds == y).sum().item()
                offline_test_correct_num += (offline_preds == y).sum().item()
                total_test_correct_num += (fused_preds == y).sum().item()


                test_num += y.shape[0]  # 将y的行数，也就是当前批次（batch）的标签数量 加到test_num中，以此来统计标签数，也就是样本数

                # —— EDIT —— 回传“在线概率”，与聚合/主监控一致
                y_prob.append(online_probs.detach().cpu().numpy())


                labels_np = y.detach().cpu().numpy()
                y_true.append(np.eye(self.num_classes)[labels_np])

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)  # 将多个训练批次的预测结果连接起来，变成一个数组（原本是一个列表，里面每一个元素都是一个训练批次的预测结果）
        y_true = np.concatenate(y_true, axis=0)  # 将多个训练批次的真实标签连接起来，变成一个数组（原本是一个列表，里面每一个元素都是一个训练批次的真实标签）

        labels_flat = np.argmax(y_true, axis=1)  # 转回标签向量

        if len(np.unique(labels_flat)) < 2:
            auc = float('nan')
        else:
            auc = metrics.roc_auc_score(labels_flat, y_prob)

        return online_test_correct_num, offline_test_correct_num, total_test_correct_num, test_num, auc, y_true, y_prob


    def train_metrics(self):
        trainloader = self.load_train_data()    #加载训练数据
        # self.model = self.load_model('model')
        # self.model.to(self.device)

        for layer, model in self.online_structure.items():
            model.eval()
        for layer, model in self.offline_structure.items():
            model.eval()

        self.model.eval()         #模型进入评估模式
        self.offline_model.eval()

        train_num = 0
        online_losses = 0
        offline_losses = 0
        total_losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)      #如果x是列表形式，那么就输出列表的第一个输入源的数据到默认设备上（默认是GPU）
                else:
                    x = x.to(self.device)      #将输入数据x移动到设备上（默认是GPU）
                y = y.to(self.device)          #将真实标签y移动到设备上（默认是GPU）

                total_online_output = None
                online_model_count = 0

                for layer, online_model in self.online_structure.items():
                    online_output = online_model(x).squeeze()
                    if online_output.dim() == 2:
                        if online_output.shape[1] == 2:
                            online_output = online_output[:, 1] - online_output[:, 0]
                        elif online_output.shape[1] == 1:
                            online_output = online_output[:, 0]
                        else:
                            raise ValueError(
                                f"Unexpected output shape {online_output.shape} for binary classification.")

                    # 确保是1D tensor
                    online_output = online_output.view(-1).to(self.device)

                    # 安全的累积方式
                    if total_online_output is None:
                        total_online_output = online_output.clone()
                    else:
                        total_online_output += online_output

                    online_model_count += 1

                # 计算平均值
                if online_model_count > 0:
                    total_online_output = total_online_output / float(online_model_count)



                total_offline_output = None
                offline_model_count = 0

                for layer, offline_model in self.offline_structure.items():
                    offline_output = offline_model(x).squeeze()
                    if offline_output.dim() == 2:
                        if offline_output.shape[1] == 2:
                            offline_output = offline_output[:, 1] - offline_output[:, 0]
                        elif offline_output.shape[1] == 1:
                            offline_output = offline_output[:, 0]
                        else:
                            raise ValueError(
                                f"Unexpected output shape {offline_output.shape} for binary classification.")

                    # 确保是1D tensor
                    offline_output = offline_output.view(-1).to(self.device)

                    # 安全的累积方式
                    if total_offline_output is None:
                        total_offline_output = offline_output.clone()
                    else:
                        total_offline_output += offline_output

                    offline_model_count += 1

                # 计算平均值
                if offline_model_count > 0:
                    total_offline_output = total_offline_output / float(offline_model_count)

                # 确保y的维度正确
                y_targets = y.float().view(-1)
                # 检查并修正batch size不匹配问题
                if total_online_output.size(0) != y_targets.size(0):
                    # 如果batch size不匹配，截取或填充到匹配的大小
                    min_batch_size = min(total_online_output.size(0), y_targets.size(0))
                    total_online_output = total_online_output[:min_batch_size]
                    y_targets = y_targets[:min_batch_size]

                online_loss = self.loss(total_online_output, y_targets)

                # 同样的检查和修正
                if total_offline_output.size(0) != y_targets.size(0):
                    min_batch_size = min(total_offline_output.size(0), y_targets.size(0))
                    total_offline_output = total_offline_output[:min_batch_size]
                    # y_targets已经在上面处理过了，这里需要重新获取
                    y_targets_offline = y.float().view(-1)[:min_batch_size]
                else:
                    y_targets_offline = y_targets
                offline_loss = self.offline_loss(total_offline_output, y_targets_offline)


                # 根据loss动态分配online和offline输出的权重占比
                ##################################################################################################################################################################
                # # 这里的online_weight、offline_weight是用于监测在测试集上的表现，self.online_weight等带self的则是在训练集上的表现
                # online_weight, offline_weight = self.compute_dynamic_weights(online_loss, offline_loss)
                # total_output = online_weight * online_output.detach() + offline_weight * offline_output.detach()
                # if self.id == 1:
                #     print("Client", self.id,
                #           " Test metrics - Online_loss:", online_loss.item(),
                #           " Offline_loss:", offline_loss.item(),
                #           " Online_weight:", online_weight.item(),
                #           " Offline_weight:", offline_weight.item())
                ##################################################################################################################################################################

                # 后续考虑根据层级递增逐渐增加offline输出占比，降低online输出占比
                ##################################################################################################################################################################
                # total_output = (total_online_output + total_offline_output) / 2
                # total_loss = self.loss(total_output,y)
                total_output = (total_online_output + total_offline_output) / 2
                total_loss = self.loss(total_output, y.float())


                online_losses += online_loss.item() * y.shape[0]
                offline_losses += offline_loss.item() * y.shape[0]
                total_losses += total_loss.item() * y.shape[0]

                train_num += y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return online_losses, offline_losses, total_losses, train_num

    def compute_dynamic_weights(self, online_loss, offline_loss):

        total_loss = online_loss.detach() + offline_loss.detach()
        online_weight_temp = online_loss / total_loss
        online_weight = 1 - online_weight_temp
        offline_weight_temp = offline_loss / total_loss
        offline_weight = 1 - offline_weight_temp



        return online_weight, offline_weight

    # —— INSERT —— EMA 更新与蒸馏缓启系数
    def _ema_update(self, teacher_model, student_model, tau=None):
        teacher_model.to(self.device)
        student_model.to(self.device)
        tau = self.ema_tau if tau is None else tau
        with torch.no_grad():
            t_params = dict(teacher_model.named_parameters())
            for n, p_s in student_model.named_parameters():
                p_t = t_params[n]
                p_t.data.mul_(tau).add_(p_s.detach().data, alpha=(1.0 - tau))

    # —— INSERT —— EMA 更新与蒸馏缓启系数
    def _kd_ramp(self, global_round_idx):
        r = (global_round_idx + 1) / max(1, self.kd_warmup_rounds)
        return float(min(1.0, max(0.0, r)))


#######################################################################################################################################################
    def train_layer_specific(self, layer, round_idx, num_clients):
        """改进的层级特定训练方法"""
        self.model.to(self.device)
        self.offline_model.to(self.device)

        # 确保结构化模型也在正确设备上
        for layer_id, model in self.online_structure.items():
            model.to(self.device)
        for layer_id, model in self.offline_structure.items():
            model.to(self.device)

        # 确保EMA教师模型也在正确设备上
        if hasattr(self, 'online_teacher'):
            self.online_teacher.to(self.device)
        if hasattr(self, 'offline_teacher'):
            self.offline_teacher.to(self.device)

        # 【新增】基于客户端欺诈特征的自适应学习率
        fraud_count = self.get_local_fraud_count()
        base_lr = self.learning_rate

        if fraud_count == 0:
            # 无欺诈样本：使用更保守的学习率
            adaptive_lr = base_lr * 0.3
            print(f"客户端 {self.id}: 无欺诈样本，使用保守学习率 {adaptive_lr:.6f}")
        elif fraud_count > 1000:
            # 大量欺诈样本：使用稍高的学习率
            adaptive_lr = base_lr * 1.2
            print(f"客户端 {self.id}: 高欺诈样本({fraud_count})，使用积极学习率 {adaptive_lr:.6f}")
        else:
            adaptive_lr = base_lr

        # 创建自适应优化器
        layer_focused_optimizer = self.create_adaptive_layer_optimizer(layer, adaptive_lr)
        original_optimizer = self.optimizer
        self.optimizer = layer_focused_optimizer

        try:
            if layer == 0:
                self.train(round_idx, num_clients)
            else:
                # self.train_with_improved_layer_fusion(layer, round_idx, num_clients)
                self.train(round_idx,num_clients)

        except Exception as e:
            print(f"客户端 {self.id} 在第 {layer} 层训练时出错: {e}")

        finally:
            self.optimizer = original_optimizer

    def get_local_fraud_count(self):
        """获取本地欺诈样本数量"""
        if hasattr(self, 'local_fraud_count'):
            return self.local_fraud_count

        # 如果没有缓存，快速计算一次
        fraud_counts = {
            0: 0, 1: 2000, 2: 718, 3: 0, 4: 1124, 5: 2452, 6: 2080, 7: 89,
            8: 0, 9: 0, 10: 297, 11: 0, 12: 0, 13: 114, 14: 497, 15: 0,
            16: 0, 17: 5186, 18: 946, 19: 5160
        }
        self.local_fraud_count = fraud_counts.get(self.id, 0)
        return self.local_fraud_count

    def create_adaptive_layer_optimizer(self, current_layer, adaptive_lr):
        """创建自适应层级优化器"""
        # 简化版本，直接使用调整后的学习率
        return torch.optim.Adam(
            self.model.parameters(),
            lr=adaptive_lr,
            weight_decay=1e-4 if self.get_local_fraud_count() > 0 else 1e-3
        )

    def train_with_improved_layer_fusion(self, layer, round_idx, num_clients):
        """改进的层级融合训练"""
        trainloader = self.load_train_data()
        self.model.train()
        self.offline_model.train()

        start_time = time.time()
        total_loss = 0.0
        batch_count = 0

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 当前层输出
                online_feature = self.model.produce_feature(x)
                online_output = self.model.classifier(online_feature).squeeze(1)
                offline_feature = self.offline_model.produce_feature(x)
                offline_output = self.offline_model.classifier(offline_feature).squeeze(1)

                # 【改进】基于理论的层级融合
                if layer > 0:
                    fused_online, fused_offline = self.compute_theory_based_layer_fusion(x, layer)

                    # 【新增】基于统计的自适应权重
                    current_stats = self.compute_output_statistics(online_output, offline_output)
                    historical_stats = self.compute_output_statistics(fused_online, fused_offline)

                    current_weight, historical_weight = self.calculate_adaptive_weights(current_stats, historical_stats)

                    final_online_output = current_weight * online_output + historical_weight * fused_online
                    final_offline_output = current_weight * offline_output + historical_weight * fused_offline

                    # 监控权重分配
                    if i == 0 and epoch == 0:
                        print(
                            f"客户端 {self.id} 层 {layer}: Current权重={current_weight:.3f}, Historical权重={historical_weight:.3f}")
                        if historical_weight > 0.7:
                            print(f"  ⚠️ High historical reliance")
                else:
                    final_online_output = online_output
                    final_offline_output = offline_output

                y = y.float().view(-1).to(self.device)

                # 计算损失
                online_loss = self.loss(final_online_output, y)
                offline_loss = self.offline_loss(final_offline_output, y)

                # 参数更新
                self.optimizer.zero_grad()
                online_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.offline_optimizer.zero_grad()
                offline_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=1.0)
                self.offline_optimizer.step()

                # EMA更新
                self._ema_update(self.online_teacher, self.model)
                self._ema_update(self.offline_teacher, self.offline_model)

                total_loss += online_loss.item() + offline_loss.item()
                batch_count += 1

        # 记录训练损失用于相似度计算
        if batch_count > 0:
            self.current_train_loss = total_loss / batch_count

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time



    def compute_output_statistics(self, output1, output2):
        """计算输出的统计特征"""
        with torch.no_grad():
            combined = torch.cat([output1, output2])
            return torch.mean(combined).item(), torch.std(combined).item()

    def calculate_adaptive_weights(self, current_stats, historical_stats):
        """基于统计特征计算自适应权重"""
        current_mean, current_std = current_stats
        historical_mean, historical_std = historical_stats

        # 如果当前输出稳定且合理，增加当前权重
        if abs(current_mean) < 2.0 and current_std < 1.0:
            current_weight = 0.8
        elif abs(current_mean) > 5.0 or current_std > 3.0:
            # 当前输出不稳定，依赖历史
            current_weight = 0.2
        else:
            current_weight = 0.6

        historical_weight = 1.0 - current_weight
        return current_weight, historical_weight

    def compute_theory_based_layer_fusion(self, x, current_layer):
        """基于理论的层级融合输出计算"""
        fused_online = torch.zeros(x.size(0), device=self.device)
        fused_offline = torch.zeros(x.size(0), device=self.device)

        total_weight = 0.0

        for layer_id in range(current_layer):
            if layer_id in self.online_structure and layer_id in self.offline_structure:
                with torch.no_grad():
                    online_model = self.online_structure[layer_id]
                    offline_model = self.offline_structure[layer_id]

                    layer_online_out = online_model(x).squeeze()
                    layer_offline_out = offline_model(x).squeeze()

                    # 处理输出维度
                    layer_online_out = self.process_model_output(layer_online_out)
                    layer_offline_out = self.process_model_output(layer_offline_out)

                    # 层级权重：更深的层权重更高
                    layer_weight = (layer_id + 1) / current_layer

                    # 检查数值稳定性
                    if (torch.abs(layer_online_out).max() < 10 and
                            torch.abs(layer_offline_out).max() < 10):
                        fused_online += layer_weight * layer_online_out
                        fused_offline += layer_weight * layer_offline_out
                        total_weight += layer_weight

        # 归一化
        if total_weight > 0:
            fused_online /= total_weight
            fused_offline /= total_weight

        return fused_online, fused_offline

    def train_with_layer_fusion(self, layer, round_idx, num_clients):
        """基于研究的层级融合训练"""
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.offline_model.to(self.device)

        # 确保历史结构模型在正确设备上
        for layer_id in self.online_structure:
            self.online_structure[layer_id].to(self.device)
        for layer_id in self.offline_structure:
            self.offline_structure[layer_id].to(self.device)

        self.model.train()
        self.offline_model.train()
        start_time = time.time()

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 当前层输出
                online_feature = self.model.produce_feature(x)
                online_output = self.model.classifier(online_feature).squeeze(1)
                offline_feature = self.offline_model.produce_feature(x)
                offline_output = self.offline_model.classifier(offline_feature).squeeze(1)

                # 使用研究基础的自适应融合
                if layer > 0:
                    fused_online_output, fused_offline_output = self.compute_layer_fusion_output(x, layer)

                    # 【新增】自适应权重计算
                    adaptive_weights = self.compute_adaptive_fusion_weights(
                        online_output, offline_output, fused_online_output, fused_offline_output
                    )

                    final_online_output = (adaptive_weights['current'] * online_output +
                                           adaptive_weights['historical'] * fused_online_output)
                    final_offline_output = (adaptive_weights['current'] * offline_output +
                                            adaptive_weights['historical'] * fused_offline_output)

                    # 监控融合效果
                    if i == 0 and epoch == 0:
                        self.log_research_based_fusion_effects(
                            layer, online_output, offline_output,
                            fused_online_output, fused_offline_output, adaptive_weights
                        )
                else:
                    final_online_output = online_output
                    final_offline_output = offline_output

                y = y.float().view(-1).to(self.device)

                # 计算损失
                online_base_loss = self.loss(final_online_output, y)
                offline_base_loss = self.offline_loss(final_offline_output, y)

                # 基于研究的一致性损失
                if layer > 0:
                    # 为online损失计算一致性损失
                    online_consistency_loss = self.compute_online_consistency_loss(
                        online_output, offline_output.detach(), fused_online_output, fused_offline_output.detach()
                    )

                    # 为offline损失计算一致性损失（重新计算融合输出避免计算图冲突）
                    fused_online_output_detached, fused_offline_output_for_offline = self.compute_layer_fusion_output(x,
                                                                                                                      layer)
                    offline_consistency_loss = self.compute_offline_consistency_loss(
                        online_output.detach(), offline_output, fused_online_output_detached.detach(),
                        fused_offline_output_for_offline
                    )

                    # 动态一致性权重
                    consistency_weight = 0.05 + 0.15 * (layer / self.max_layers if hasattr(self, 'max_layers') else 0.1)

                    online_loss = online_base_loss + consistency_weight * online_consistency_loss
                    offline_loss = offline_base_loss + consistency_weight * offline_consistency_loss
                else:
                    online_loss = online_base_loss
                    offline_loss = offline_base_loss

                # 参数更新
                self.optimizer.zero_grad()
                online_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.offline_optimizer.zero_grad()
                offline_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=1.0)
                self.offline_optimizer.step()

                # EMA更新
                self._ema_update(self.online_teacher, self.model)
                self._ema_update(self.offline_teacher, self.offline_model)

        # 异常检测
        self.model.eval()
        self.offline_model.eval()

        with torch.no_grad():
            test_loader = self.load_train_data()
            test_batch_x, test_batch_y = next(iter(test_loader))

            if type(test_batch_x) == type([]):
                test_batch_x[0] = test_batch_x[0].to(self.device)
            else:
                test_batch_x = test_batch_x.to(self.device)
            test_batch_y = test_batch_y.to(self.device)

            online_feature = self.model.produce_feature(test_batch_x)
            online_output = self.model.classifier(online_feature).squeeze(1)
            offline_feature = self.offline_model.produce_feature(test_batch_x)
            offline_output = self.offline_model.classifier(offline_feature).squeeze(1)

            if layer > 0:
                fused_online_output, fused_offline_output = self.compute_layer_fusion_output(test_batch_x, layer)
                adaptive_weights = self.compute_adaptive_fusion_weights(
                    online_output, offline_output, fused_online_output, fused_offline_output
                )
                final_online_output = (adaptive_weights['current'] * online_output +
                                       adaptive_weights['historical'] * fused_online_output)
                final_offline_output = (adaptive_weights['current'] * offline_output +
                                        adaptive_weights['historical'] * fused_offline_output)
            else:
                final_online_output = online_output
                final_offline_output = offline_output

            test_batch_y = test_batch_y.float().view(-1).to(self.device)

            online_loss = self.loss(final_online_output, test_batch_y).item()
            offline_loss = self.offline_loss(final_offline_output, test_batch_y).item()

            online_pred = torch.sigmoid(final_online_output) > 0.5
            offline_pred = torch.sigmoid(final_offline_output) > 0.5

            online_acc = (online_pred == test_batch_y).float().mean().item()
            offline_acc = (offline_pred == test_batch_y).float().mean().item()

            anomaly = self.check_anomaly(online_loss, offline_loss, online_acc, offline_acc)

            if anomaly:
                print(f"[CLIENT {self.id}] Layer {layer} - Round {round_idx} ⚠️ ANOMALY")
                print(f"  Online: Loss={online_loss:.3f}, Acc={online_acc:.3f}")
                print(f"  Offline: Loss={offline_loss:.3f}, Acc={offline_acc:.3f}")

        self.model.train()
        self.offline_model.train()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def compute_adaptive_fusion_weights(self, current_online, current_offline, fused_online, fused_offline):
        """基于输出质量的自适应融合权重"""
        with torch.no_grad():
            # 计算输出统计特征
            current_online_std = torch.std(current_online).item()
            current_offline_std = torch.std(current_offline).item()
            fused_online_std = torch.std(fused_online).item()
            fused_offline_std = torch.std(fused_offline).item()

            # 计算相对贡献度
            online_contrib = abs(torch.mean(fused_online).item()) / (abs(torch.mean(current_online).item()) + 1e-8)
            offline_contrib = abs(torch.mean(fused_offline).item()) / (abs(torch.mean(current_offline).item()) + 1e-8)

            # 自适应权重策略
            if online_contrib < 0.01 and offline_contrib < 0.01:
                # 历史贡献极小
                historical_weight = 0.05
            elif current_online_std > fused_online_std * 3:
                # 当前输出不稳定，增加历史权重
                historical_weight = 0.6
            else:
                # 正常情况，基于贡献度调整
                historical_weight = 0.2 + 0.3 * min(online_contrib, offline_contrib)

            current_weight = 1.0 - historical_weight

            return {
                'current': current_weight,
                'historical': historical_weight
            }

    def compute_research_based_consistency_loss(self, online_output, offline_output, fused_online, fused_offline):
        """基于研究的一致性损失"""
        # 特征级一致性
        feature_consistency = F.mse_loss(online_output, offline_output.detach())

        # 融合一致性
        fusion_consistency_online = F.mse_loss(online_output, fused_online.detach())
        fusion_consistency_offline = F.mse_loss(offline_output, fused_offline.detach())

        return (feature_consistency + fusion_consistency_online + fusion_consistency_offline) / 3

    def log_research_based_fusion_effects(self, layer, online_output, offline_output,
                                          fused_online, fused_offline, adaptive_weights):
        """基于研究的融合效果监控"""
        with torch.no_grad():
            online_stats = {
                'mean': torch.mean(online_output).item(),
                'std': torch.std(online_output).item(),
                'range': (torch.min(online_output).item(), torch.max(online_output).item())
            }

            fused_stats = {
                'mean': torch.mean(fused_online).item(),
                'std': torch.std(fused_online).item(),
                'range': (torch.min(fused_online).item(), torch.max(fused_online).item())
            }

            contribution_ratio = abs(fused_stats['mean']) / (
                        abs(online_stats['mean']) + abs(fused_stats['mean']) + 1e-8)

            print(f"Client {self.id} Layer {layer} [Research-Based Analysis]:")
            print(f"  Current: mean={online_stats['mean']:.3f}, std={online_stats['std']:.3f}")
            print(f"  Historical: mean={fused_stats['mean']:.3f}, std={fused_stats['std']:.3f}")
            print(
                f"  Adaptive weights: Current={adaptive_weights['current']:.3f}, Historical={adaptive_weights['historical']:.3f}")
            print(f"  Historical contribution: {contribution_ratio:.3f}")

            # 基于研究的诊断
            if contribution_ratio < 0.01:
                print(f"  ✓ Minimal historical impact - appropriate weight reduction")
            elif adaptive_weights['historical'] > 0.5:
                print(f"  ⚠️  High historical reliance - check current layer stability")

    def compute_layer_fusion_output(self, x, current_layer):
        """基于FuseFL理论的渐进式层级融合"""
        fused_online = torch.zeros(x.size(0), device=self.device)
        fused_offline = torch.zeros(x.size(0), device=self.device)

        # 【FuseFL核心】按自底向上方式渐进融合
        layer_contributions_online = []
        layer_contributions_offline = []
        layer_weights = []

        # 收集所有历史层输出
        for layer_id in range(current_layer):
            if layer_id in self.online_structure and layer_id in self.offline_structure:
                with torch.no_grad():
                    online_model = self.online_structure[layer_id]
                    offline_model = self.offline_structure[layer_id]

                    layer_online_out = online_model(x).squeeze()
                    layer_offline_out = offline_model(x).squeeze()

                    # 处理维度（保持原有逻辑）
                    layer_online_out = self.process_model_output(layer_online_out)
                    layer_offline_out = self.process_model_output(layer_offline_out)

                    layer_contributions_online.append(layer_online_out)
                    layer_contributions_offline.append(layer_offline_out)

                    # 【FuseFL核心】层级重要性权重：越深层权重越大
                    layer_importance = (layer_id + 1) / current_layer
                    layer_weights.append(layer_importance)

        # 【新增】数值稳定性检查和归一化
        if layer_contributions_online:
            normalized_weights = self.normalize_layer_weights(
                layer_weights, layer_contributions_online, layer_contributions_offline
            )

            # 加权融合
            for online_out, offline_out, weight in zip(
                    layer_contributions_online, layer_contributions_offline, normalized_weights
            ):
                fused_online += weight * online_out
                fused_offline += weight * offline_out

        return fused_online, fused_offline

    def process_model_output(self, model_output):
        """处理模型输出维度"""
        if model_output.dim() == 2:
            if model_output.shape[1] == 2:
                return model_output[:, 1] - model_output[:, 0]
            elif model_output.shape[1] == 1:
                return model_output[:, 0]
        return model_output

    def normalize_layer_weights(self, raw_weights, online_outputs, offline_outputs):
        """基于输出统计的权重归一化"""
        normalized_weights = []

        for i, (weight, online_out, offline_out) in enumerate(
                zip(raw_weights, online_outputs, offline_outputs)
        ):
            # 计算输出的数值特征
            online_magnitude = torch.abs(online_out).mean().item()
            offline_magnitude = torch.abs(offline_out).mean().item()

            # 数值过小的层降低权重
            if online_magnitude < 0.01 and offline_magnitude < 0.01:
                adjusted_weight = weight * 0.1
            elif online_magnitude > 10 or offline_magnitude > 10:
                # 数值过大的层也降低权重
                adjusted_weight = weight * 0.5
            else:
                adjusted_weight = weight

            normalized_weights.append(adjusted_weight)

        # 归一化权重和
        total_weight = sum(normalized_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in normalized_weights]

        return normalized_weights

    def compute_layer_consistency_loss(self, current_online, current_offline, fused_online, fused_offline):
        """计算层级间一致性损失"""
        # 计算当前层输出与融合输出的一致性
        online_consistency = F.mse_loss(current_online, fused_online.detach())
        offline_consistency = F.mse_loss(current_offline, fused_offline.detach())
        return (online_consistency + offline_consistency) / 2

    def compute_relation_distillation(self, online_features, offline_features):
        """特征层面的关系蒸馏"""
        batch_size = online_features.size(0)

        # 批次太小时跳过关系蒸馏
        if batch_size < 2:
            return torch.tensor(0.0, device=online_features.device, requires_grad=True)

        try:
            # 计算特征间的欧氏距离矩阵
            online_dist = torch.cdist(online_features, online_features, p=2)
            offline_dist = torch.cdist(offline_features, offline_features, p=2)

            # 归一化距离矩阵以提高稳定性
            online_dist = online_dist / (online_dist.max() + 1e-8)
            offline_dist = offline_dist / (offline_dist.max() + 1e-8)

            # 关系一致性损失 - 使用detach()避免梯度循环
            relation_loss = F.mse_loss(online_dist, offline_dist.detach()) + \
                            F.mse_loss(offline_dist, online_dist.detach())

            return relation_loss * 0.5  # 取平均

        except Exception as e:
            print(f"关系蒸馏计算出错: {e}")
            return torch.tensor(0.0, device=online_features.device, requires_grad=True)

    def create_layer_focused_optimizer(self, current_layer, base_lr=None):
        """创建层级专注的优化器 - 使用渐进式学习率策略"""
        if base_lr is None:
            base_lr = self.learning_rate

        if not hasattr(self.model, 'get_layer_parameters'):
            # 如果是旧架构，使用标准优化器
            return torch.optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=1e-5)

        layer_params = self.model.get_layer_parameters()
        param_groups = []

        for layer_idx, layer_modules in enumerate(layer_params):
            # 收集该层的所有参数
            layer_param_list = []
            for module in layer_modules:
                if isinstance(module, nn.Module):
                    layer_param_list.extend(list(module.parameters()))
                elif hasattr(module, '__iter__'):
                    for m in module:
                        if isinstance(m, nn.Module):
                            layer_param_list.extend(list(m.parameters()))

            # 设置学习率
            if layer_idx < current_layer:
                lr = base_lr * 0.01  # 已训练层：很小的学习率
            elif layer_idx == current_layer:
                lr = base_lr  # 当前训练层：正常学习率
            else:
                lr = base_lr * 0.1  # 未来层：中等学习率

            if hasattr(self, 'anomaly_count') and self.anomaly_count > 3:
                lr *= 0.5
                if layer_idx == current_layer:  # 只在当前层打印日志
                    print(
                        f"Client {self.id}: Reducing LR for layer {current_layer} due to {self.anomaly_count} anomalies")

            if layer_param_list:
                param_groups.append({'params': layer_param_list, 'lr': lr})

        return torch.optim.Adam(param_groups, weight_decay=1e-5)

    def apply_gradual_learning_rates(self, optimizer, current_layer):
        """为现有优化器应用渐进式学习率"""
        if not hasattr(self.model, 'get_layer_parameters'):
            return optimizer

        layer_params = self.model.get_layer_parameters()
        base_lr = self.learning_rate

        # 重新设置参数组的学习率
        param_idx = 0
        for layer_idx, layer_modules in enumerate(layer_params):
            # 计算该层参数数量
            layer_param_count = 0
            for module in layer_modules:
                if isinstance(module, nn.Module):
                    layer_param_count += len(list(module.parameters()))
                elif hasattr(module, '__iter__'):
                    for m in module:
                        if isinstance(m, nn.Module):
                            layer_param_count += len(list(m.parameters()))

            # 设置学习率
            if layer_idx < current_layer:
                lr = base_lr * 0.01  # 已训练层：很小的学习率
            elif layer_idx == current_layer:
                lr = base_lr  # 当前训练层：正常学习率
            else:
                lr = base_lr * 0.1  # 未来层：中等学习率

            # 更新优化器中对应参数组的学习率
            for _ in range(layer_param_count):
                if param_idx < len(optimizer.param_groups):
                    optimizer.param_groups[param_idx]['lr'] = lr
                    param_idx += 1

        return optimizer

    def check_anomaly(self, online_loss, offline_loss, online_acc, offline_acc):
        # 根据训练轮数动态调整阈值
        if hasattr(self, 'round_count'):
            # 早期训练：宽松阈值
            if self.round_count < 10:
                loss_threshold = 3.0
            # 中期训练：中等阈值
            elif self.round_count < 50:
                loss_threshold = 2.0
            # 后期训练：严格阈值
            else:
                loss_threshold = 1.0
        else:
            loss_threshold = 2.0

        # 基于历史损失的相对阈值
        if hasattr(self, 'loss_history') and len(self.loss_history) >= 5:
            avg_loss = np.mean(self.loss_history[-5:])
            std_loss = np.std(self.loss_history[-5:])
            relative_threshold = avg_loss + 2 * std_loss  # 2倍标准差
            loss_threshold = min(loss_threshold, relative_threshold)

    def compute_online_consistency_loss(self, online_output, offline_output_detached, fused_online,
                                        fused_offline_detached):
        """为online模型计算一致性损失"""
        # 特征级一致性
        feature_consistency = F.mse_loss(online_output, offline_output_detached)

        # 融合一致性 - 只包含online相关的计算图
        fusion_consistency = F.mse_loss(online_output, fused_online)

        return (feature_consistency + fusion_consistency) / 2

    def compute_offline_consistency_loss(self, online_output_detached, offline_output, fused_online_detached,
                                         fused_offline):
        """为offline模型计算一致性损失"""
        # 特征级一致性
        feature_consistency = F.mse_loss(offline_output, online_output_detached)

        # 融合一致性 - 只包含offline相关的计算图
        fusion_consistency = F.mse_loss(offline_output, fused_offline)

        return (feature_consistency + fusion_consistency) / 2


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

    def analyze_data_locally(self,train_loader):
        """完全本地化的数据分析 - 无隐私泄露"""
        try:
            # train_loader = self.load_train_data()
            positive_count = 0
            total_count = 0

            for x, y in train_loader:
                positive_count += y.sum().item()
                total_count += len(y)

            self.local_fraud_rate = positive_count / total_count if total_count > 0 else 0

            # 仅基于本地数据调整学习策略，不向外传输任何信息
            if self.local_fraud_rate > 0.15:
                # 高欺诈率：降低学习率，增加正则化
                self.learning_rate *= 0.8
                self.client_adaptation = "conservative"
            elif self.local_fraud_rate < 0.02:
                # 低欺诈率：可能需要更敏感的学习
                self.learning_rate *= 1.1
                self.client_adaptation = "sensitive"
            else:
                self.client_adaptation = "standard"

            # 只在本地记录，不传输
            print(f"Client {self.id}: Local adaptation = {self.client_adaptation}")

        except Exception as e:
            print(f"Client {self.id}: Local analysis failed: {e}")
            self.local_fraud_rate = 0.05
            self.client_adaptation = "standard"

    def log_fusion_effects(self, layer, online_output, offline_output, fused_online, fused_offline, batch_idx):
        """监控融合效果"""
        if not hasattr(self, 'fusion_logs'):
            self.fusion_logs = []

        with torch.no_grad():
            # 计算当前层输出的统计信息
            online_mean = torch.mean(online_output).item()
            online_std = torch.std(online_output).item()

            # 计算融合输出的统计信息
            fused_mean = torch.mean(fused_online).item()
            fused_std = torch.std(fused_online).item()

            # 计算融合对最终输出的影响
            final_before_fusion = online_output
            final_after_fusion = 0.6 * online_output + 0.4 * fused_online

            fusion_impact = torch.mean(torch.abs(final_after_fusion - final_before_fusion)).item()

            # 计算历史层的贡献度
            contribution_ratio = abs(fused_mean) / (abs(online_mean) + abs(fused_mean) + 1e-8)

            log_entry = {
                'layer': layer,
                'batch': batch_idx,
                'online_mean': online_mean,
                'fused_mean': fused_mean,
                'fusion_impact': fusion_impact,
                'contribution_ratio': contribution_ratio,
                'weight_effectiveness': fusion_impact / 0.4  # 0.4是融合权重
            }

            self.fusion_logs.append(log_entry)

            # 每个层级每轮只分析一次
            if batch_idx == 0:
                print(f"Client {self.id} Layer {layer}: Fusion impact={fusion_impact:.4f}, "
                      f"Historical contribution={contribution_ratio:.3f}")

                # 如果融合影响过小，可能权重设置有问题
                if fusion_impact < 0.001:
                    print(f"  Warning: Very low fusion impact - consider adjusting weights")
                elif fusion_impact > 0.5:
                    print(f"  Warning: Very high fusion impact - may cause instability")

    def apply_fedprox_proximal(self, model, global_params_dict, mu):
        """
        Compute proximal penalty: (mu/2) * sum ||param - global_param||^2
        global_params_dict: mapping name -> tensor (server-sent snapshot)
        """
        if mu is None or mu <= 0:
            return 0.0
        prox = 0.0
        for name, p in model.named_parameters():
            if name in global_params_dict:
                g = global_params_dict[name].to(p.device)
                prox += ((p - g).norm(2) ** 2)
        return 0.5 * mu * prox

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