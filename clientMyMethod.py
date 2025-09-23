
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



        # 离线模型：加权BCE，给正样本更高权重
        def get_weighted_bce():
            def weighted_bce_loss(inputs, targets):
                pos_count = (targets == 1).sum()
                neg_count = (targets == 0).sum()

                if pos_count > 0:
                    pos_weight = neg_count.float() / pos_count.float()
                    # pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)  # 限制权重范围
                    pos_weight = torch.clamp(pos_weight, min=1.0,max=50.0)  # 不限制权重范围
                else:
                    pos_weight = torch.tensor(1.0)

                return F.binary_cross_entropy_with_logits(
                    inputs, targets,
                    pos_weight=pos_weight,
                    reduction='mean'
                )

            return weighted_bce_loss

        self.loss = get_weighted_bce()
        self.offline_loss = get_weighted_bce()

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

        self.relation_kd_weight = getattr(args, 'relation_kd_weight', 0.1)
        self.relation_kd_enabled = getattr(args, 'relation_kd_enabled', True)

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






    def train(self,global_rounds,num_clients): #已经废弃！！！！！！！！！！！！！！！！！

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

    def train_layered(self, layer_idx, round_idx, num_clients, alpha):
        """
        在线模型（教师）与离线模型（学生）协同训练的函数。
        两个模型共同优化，实现泛化知识与本地知识的融合。
        """

        # -------------------------------------------------------------------
        # 步骤 1: 设置模型模式、设备和优化器
        # -------------------------------------------------------------------
        trainloader = self.load_train_data()

        # 两个模型都设置为训练模式，因为它们都将被更新
        self.model.train()  # 在线模型
        self.offline_model.train()  # 离线模型
        self.model.to(self.device)
        self.offline_model.to(self.device)

        # 在这个协同训练方案中，我们更新整个模型，而不是特定的层级
        # 确保优化器是为整个模型设置的
        # 这里我们保留你原有的双优化器设置

        # -------------------------------------------------------------------
        # 步骤 2: 训练循环和损失计算
        # -------------------------------------------------------------------
        epoch_losses = []

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

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

                # 教师模型（在线模型）前向传播
                # 这里的在线模型是教师，但它也会被更新，所以不使用 torch.no_grad()
                teacher_output = self.model(x)

                # 学生模型（离线模型）前向传播
                student_output = self.offline_model(x)

                # 统一输出维度
                if student_output.dim() > 1:
                    student_output = student_output.squeeze()
                if teacher_output.dim() > 1:
                    teacher_output = teacher_output.squeeze()

                local_train_online_loss = self.loss(teacher_output, y)
                local_train_offline_loss = self.offline_loss(student_output, y)

                if round_idx != 0:
                    # 计算知识蒸馏损失
                    # 学生（离线）向教师（在线）学习
                    kd_loss_offline_to_online = self.compute_kd_loss(student_output, teacher_output, y, alpha)

                    # 额外增加一个反向的蒸馏损失，让在线模型也从离线模型的本地知识中学习
                    kd_loss_online_to_offline = self.compute_kd_loss(teacher_output, student_output, y, alpha)

                    # 组合总损失
                    # 在线模型总损失 = 硬标签损失 + 反向蒸馏损失
                    total_online_loss = local_train_online_loss + kd_loss_online_to_offline
                    # 离线模型总损失 = 硬标签损失 + 蒸馏损失
                    total_offline_loss = local_train_offline_loss + kd_loss_offline_to_online

                # -------------------------------------------------------------------
                # 步骤 3: 损失组合和反向传播
                # -------------------------------------------------------------------
                else:
                    # 组合总损失
                    # 在线模型总损失 = 硬标签损失 + 反向蒸馏损失
                    total_online_loss = local_train_online_loss
                    # 离线模型总损失 = 硬标签损失 + 蒸馏损失
                    total_offline_loss = local_train_offline_loss

                # 应用 FedProx 或其他正则化
                mu = getattr(self, 'fedprox_mu', 0.0)
                if mu > 0:
                    global_snapshot = getattr(self, 'initial_params', None)
                    offline_snapshot = getattr(self, 'initial_offline_params', None)
                    if global_snapshot is not None:
                        prox_online = self.apply_fedprox_proximal(self.model, global_snapshot, mu)
                        total_online_loss += prox_online
                    if offline_snapshot is not None:
                        prox_offline = self.apply_fedprox_proximal(self.offline_model, offline_snapshot, mu)
                        total_offline_loss += prox_offline

                # -------------------------------------------------------------------
                # 步骤 4: 优化两个模型
                # -------------------------------------------------------------------
                #考虑onlineloss和offlineloss权重？-----------------------------------------------------------------
                total_loss = total_online_loss + total_offline_loss
                self.optimizer.zero_grad()
                self.offline_optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=1.0)
                # 执行优化步骤
                self.optimizer.step()
                self.offline_optimizer.step()

                # # 优化在线模型
                # self.optimizer.zero_grad()
                # total_online_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # self.optimizer.step()
                #
                # # 优化离线模型
                # self.offline_optimizer.zero_grad()
                # total_offline_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.offline_model.parameters(), max_norm=1.0)
                # self.offline_optimizer.step()

                batch_losses.append(total_online_loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

            # -------------------------------------------------------------------
            # 步骤 5: 更新历史信息
            # -------------------------------------------------------------------
        self.online_structure[layer_idx] = copy.deepcopy(self.model).state_dict()
        self.offline_structure[layer_idx] = copy.deepcopy(self.offline_model).state_dict()
        # 在这个方案中，两个模型都通过优化器更新，所以不需要动量更新

        # 计算平均训练损失并更新历史信息
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.training_loss = avg_loss
        global_loss = getattr(self, 'received_global_loss', self.global_loss)
        self.update_lr_history(avg_loss, global_loss, round_idx)

        print(f"[CLIENT {self.id}] 协同训练完成, 平均损失: {avg_loss:.4f}")



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

        # for key in self.online_structure.keys():
        #     self.online_structure[key].to(self.device)
        # for key in self.offline_structure.keys():
        #     self.offline_structure[key].to(self.device)

    def compute_consistency_loss(self, online_features, offline_features):
        """添加特征一致性损失"""
        return F.mse_loss(online_features, offline_features.detach())



    def compute_dynamic_weights(self, online_loss, offline_loss):

        total_loss = online_loss.detach() + offline_loss.detach()
        online_weight_temp = online_loss / total_loss
        online_weight = 1 - online_weight_temp
        offline_weight_temp = offline_loss / total_loss
        offline_weight = 1 - offline_weight_temp



        return online_weight, offline_weight




#######################################################################################################################################################



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
#----------------------------------------------------------------------------------------------------------------
    def update_teacher_momentum(self):
        """动量更新Teacher模型"""
        self.teacher_model.to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            for teacher_param, student_param in zip(
                    self.teacher_model.parameters(),
                    self.model.parameters()
            ):
                teacher_param.data = (
                        self.momentum_tau * teacher_param.data +
                        (1 - self.momentum_tau) * student_param.data
                )

    def compute_kd_loss(self, student_logits, teacher_logits, labels, alpha):
        """计算单向知识蒸馏损失"""
        # 硬标签损失
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

        # 软标签损失（KL散度）
        student_probs = torch.sigmoid(student_logits / self.temperature)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)

        soft_loss = F.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 组合损失
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
        return total_loss

    def train_layer_specific(self, layer_idx, round_idx, num_clients, alpha):#已经废弃！！！！！！！！！！！！！！！！！
        """单向teacher-student架构的层级训练"""
        trainloader = self.load_train_data()
        # self.model.train()
        # self.teacher_model.eval()
        #
        # self.teacher_model.to(self.device)
        # self.model.to(self.device)

        # 确保正在训练正确的层级
        if layer_idx not in self.online_structure:
            print(f"警告: 客户端 {self.id} 在 {layer_idx} 层找不到模型")
            return

        # 获取当前层的学生和教师模型模块
        student_model_layer = self.online_structure[layer_idx]
        teacher_model_layer = self.offline_structure[layer_idx]

        student_model_layer.train()
        teacher_model_layer.eval()
        self.model.train()
        self.offline_model.eval()
        self.model.to(self.device)
        self.offline_model.to(self.device)
        student_model_layer.to(self.device)
        teacher_model_layer.to(self.device)

        # === 设置自适应学习率 ===
        optimizer = torch.optim.Adam(student_model_layer.parameters(), lr=self.get_adaptive_lr())

        trainloader = self.load_train_data()
        epoch_losses = []

        epoch_losses = []

        # 本地训练轮次
        for epoch in range(self.local_epochs):
            batch_losses = []

            for step, (x, y) in enumerate(trainloader):
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                x, y = x.to(self.device), y.to(self.device).float()
                if y.dim() > 1:
                    y = y.squeeze()
                if x.ndim > 2:
                    x = x.view(x.size(0), -1)

                with torch.no_grad():
                    # 教师模型前向传播
                    teacher_features = self.offline_model.produce_feature(x)
                    teacher_output = self.offline_model.classifier(teacher_features)

                # 学生模型前向传播
                # 如果训练的是分类器，需要先用特征提取器获取特征
                if layer_idx == 1:
                    with torch.no_grad():
                        student_features = self.model.produce_feature(x)
                    student_output = student_model_layer(student_features)
                # 如果训练的是特征提取器，直接前向传播
                else:
                    student_features = student_model_layer.produce_feature(x)
                    student_output = self.model.classifier(student_features)

                if student_output.dim() > 1:
                    student_output = student_output.squeeze()
                if teacher_output.dim() > 1:
                    teacher_output = teacher_output.squeeze()

                # 计算知识蒸馏损失
                loss = self.compute_kd_loss(student_output, teacher_output, y, alpha)

                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

            # 每个epoch结束后更新teacher
            self.update_teacher_momentum()

        # 计算平均训练损失并更新历史信息
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.training_loss = avg_loss  # 保存训练损失供服务器使用

        # 注意：global_loss需要从服务器获取，这里暂时使用1.0
        global_loss = getattr(self, 'received_global_loss', self.global_loss)
        self.update_lr_history(avg_loss, global_loss, round_idx)

        print(f"[CLIENT {self.id}] 第{layer_idx}层训练完成, 平均损失: {avg_loss:.4f}")

    def get_adaptive_lr(self):
        """基于性能自适应调整学习率"""
        # Warmup阶段
        if self.current_round < self.warmup_rounds:
            return self.base_lr * (self.current_round + 1) / self.warmup_rounds

        # 基于损失变化率调整
        if len(self.loss_history) >= 2:
            recent_change = abs(self.loss_history[-1] - self.loss_history[-2])
            if recent_change < 0.01:  # 收敛缓慢
                self.convergence_rate *= 1.1
            elif recent_change > 0.5:  # 震荡过大
                self.convergence_rate *= 0.8

        # 基于全局-本地性能差异
        performance_gap = abs(self.last_loss - self.global_loss) / (self.global_loss + 1e-8)
        gap_factor = 1.0 + 0.2 * performance_gap

        adaptive_lr = (self.base_lr *
                       self.convergence_rate *
                       gap_factor *
                       (self.decay_factor ** (self.current_round - self.warmup_rounds)))

        # 限制学习率范围
        return max(1e-5, min(0.01, adaptive_lr))

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


# ----------------------------------------------------------------------------------------------------------------