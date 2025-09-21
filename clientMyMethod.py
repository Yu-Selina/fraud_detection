
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



        self.online_structure = {}  # 存每个层级的在线模型
        self.offline_structure = {} # 存每个层级的离线模型

        self.relation_kd_weight = getattr(args, 'relation_kd_weight', 0.1)
        self.relation_kd_enabled = getattr(args, 'relation_kd_enabled', True)

        # 监控相关变量
        self.anomaly_count = 0
        self.prev_loss = None
        self.detailed_logging = False
        self.local_analysis_done = False   #训练开始前调用本地数据分析

        # 单向Teacher-Student参数
        self.momentum_tau = getattr(args, 'momentum_tau', 0.99)
        self.temperature = getattr(args, 'temperature', 4.0)
        self.alpha = getattr(args, 'distill_alpha', 0.7)

        # 自适应学习率相关参数（替代AdaptiveLRScheduler类）
        self.base_lr = args.local_learning_rate
        self.warmup_rounds = getattr(args, 'warmup_rounds', 5)
        self.decay_factor = getattr(args, 'decay_factor', 0.95)
        self.loss_history = []
        self.convergence_rate = 1.0
        self.current_round = 0
        self.last_loss = 1.0
        self.global_loss = 1.0

        # 初始化Teacher模型
        self.teacher_model = copy.deepcopy(self.model)
        for param in self.teacher_model.parameters():
            param.requires_grad = False





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
        with torch.no_grad():
            for teacher_param, student_param in zip(
                    self.teacher_model.parameters(),
                    self.model.parameters()
            ):
                teacher_param.data = (
                        self.momentum_tau * teacher_param.data +
                        (1 - self.momentum_tau) * student_param.data
                )

    def compute_kd_loss(self, student_logits, teacher_logits, labels):
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
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss

    def train_layer_specific(self, layer_idx, round_idx, num_clients):
        """单向teacher-student架构的层级训练"""
        trainloader = self.load_train_data()
        self.model.train()
        self.teacher_model.eval()

        self.teacher_model.to(self.device)

        # === 设置自适应学习率 ===
        adaptive_lr = self.get_adaptive_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adaptive_lr
        print(f"客户端 {self.id}: 自适应学习率 {adaptive_lr:.6f}")

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

                # 学生模型前向传播
                student_output = self.model(x)

                # 教师模型前向传播（无梯度）
                with torch.no_grad():
                    teacher_output = self.teacher_model(x)

                if student_output.dim() > 1:
                    student_output = student_output.squeeze()
                if teacher_output.dim() > 1:
                    teacher_output = teacher_output.squeeze()

                # 计算知识蒸馏损失
                loss = self.compute_kd_loss(student_output, teacher_output, y)

                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
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

# ----------------------------------------------------------------------------------------------------------------
