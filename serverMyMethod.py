import copy
import math
import os
import time
import random
import torch
import numpy as np

from system.flcore.clients.clientavg import clientAVG
from system.flcore.servers.serverbase import Server
from system.flcore.servers.serveravg import FedAvg
from system.flcore.clients.clientMyMethod import clientMyMethod
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, auc,  roc_auc_score, average_precision_score, confusion_matrix
import torch.nn.functional as F
from config_privacy import *
from torch.utils.data import TensorDataset, DataLoader



class MyMethod(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

        global_test_path = os.path.join(args.data_dir, "global_test_set.npz")
        if os.path.exists(global_test_path):
            data = np.load(global_test_path, allow_pickle=True)
            x = torch.tensor(data['x']).float()
            y = torch.tensor(data['y']).float()
            self.global_test_data = TensorDataset(x, y)
            print(f"成功加载全局测试集，样本数量: {len(self.global_test_data)}")
        else:
            self.global_test_data = None
            print("警告: 未找到全局测试集文件。请先运行 generate_fraud_detection.py。")

        self.classifier_list = {}
        self.rs_online_test_acc = []
        self.rs_offline_test_acc = []
        self.rs_total_test_acc = []
        self.rs_online_train_loss = []
        self.rs_offline_train_loss = []
        self.rs_total_train_loss = []
        self.rs_test_recall = []
        self.rs_test_f1 = []
        self.rs_test_auprc = []
        self.rs_test_auc = []
        self.eval_rounds = []
        self.current = 0


        self.patience = args.patience
        self.early_stop_mode = args.early_stop_mode
        self.min_delta = args.min_delta
        self.best_score = None
        self.num_bad_rounds = 0
        self.early_stop = False

        self.structure= {0:{}} #存每个层级、每个聚类的全局模型
        self.num_layers = 0

        self.max_layers = getattr(args, 'max_layers', 3)
        self.min_clients_per_cluster = getattr(args, 'min_clients_per_cluster', 2)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.7)
        self.layer_rounds = getattr(args, 'layer_rounds', [30, 20, 15])  # 每层训练轮数
        # 客户端性能跟踪
        self.client_performance_history = {}  # {client_id: [performance_scores]}
        self.active_clients_set = set(range(self.num_clients))
        # 相似度计算缓存
        self.client_data_distributions = {}
        # 隐私保护的训练统计
        self.client_model_updates = [[] for _ in range(self.num_clients)]
        self.client_training_history = [[] for _ in range(self.num_clients)]
        self.client_parameters_history = [[] for _ in range(self.num_clients)]

        self.ppo_epsilon = getattr(args, 'ppo_epsilon', 0.2)  # PPO裁剪参数
        self.old_global_params = None
        self.ppo_enabled = getattr(args, 'ppo_enabled', True)

        # 启用理论驱动的改进
        self.fraud_aware_aggregation_enabled = True
        self.gradient_conflict_detection = True
        self.theoretical_early_stopping = True

        # 为极端non-IID调整的参数
        self.min_delta = 0.001  # 更小的改善阈值
        self.patience = 10  # 更短的耐心，因为欺诈检测收敛慢

        self.global_loss = 1.0  # 用于跟踪全局损失

        self.initial_global_model = copy.deepcopy(self.global_model)

    # 修改主训练函数
    def train(self):
        """多层级训练主循环"""
        # 初始化第0层
        self.structure[0][0] = [np.arange(self.num_clients), copy.deepcopy(self.global_model)]

        for layer in range(self.max_layers):
            print(f"\n{'=' * 50}")
            print(f"开始第 {layer} 层训练")
            print(f"活跃客户端: {sorted(list(self.active_clients_set))}")
            print(f"{'=' * 50}")

            # 训练当前层
            self.train_current_layer(layer)

            # 如果不是最后一层，生成下一层聚类
            if layer < self.max_layers - 1 and len(self.active_clients_set) > self.min_clients_per_cluster:
                self.generate_next_layer_clusters(layer)
            else:
                print(f"在第 {layer} 层结束训练")
                break

        # 原有的保存和绘图代码保持不变
        self.save_results()
        self.save_global_model()
        self.plot_metrics()

    def train_current_layer(self, layer):
        """训练指定层级的联邦学习"""
        print(f"\n开始第 {layer} 层训练")
        print(f"活跃客户端: {[client.id for client in self.selected_clients]}")

        # 渐进式层级训练设置
        self.progressive_layer_training(layer)

        # 根据层级决定聚类策略
        if layer == 0:
            # 第0层使用全局聚合
            clusters = [[client.id for client in self.selected_clients]]
            print(f"第 {layer} 层采用全局聚合")
        else:
            # 高层级使用梯度相似度聚类
            client_updates = self.collect_client_updates()
            clusters = self.adaptive_layer_clustering(client_updates, layer)
            print(f"第 {layer} 层生成了 {len(clusters)} 个聚类")
            for i, cluster in enumerate(clusters):
                print(f"聚类 {i}: 客户端 {cluster}")


        layer_rounds = self.layer_rounds[layer] if layer < len(self.layer_rounds) else 15

        #初始化层级监控
        if not hasattr(self, 'layer_metrics'):
            self.layer_metrics = {}
        self.layer_metrics[layer] = {
            'gradient_conflicts': [],
            'fraud_gradient_norms': [],
            'convergence_scores': []
        }

        for round_idx in range(layer_rounds):
            s_t = time.time()

            # 选择活跃客户端
            self.selected_clients = self.select_active_clients()
            self.send_models_with_classifier(layer)

            noisy_client_ratios = self.get_noisy_pos_ratios(self.selected_clients)# 在每轮循环内部，为本轮被选中的客户端计算自适应 alpha
            adaptive_alphas = self.compute_adaptive_alpha(noisy_client_ratios)

            #梯度冲突检测
            if round_idx > 0:
                conflict_detected = self.detect_fraud_gradient_conflicts(layer)
                if conflict_detected:
                    self.adjust_aggregation_strategy(layer, "conflict_mode")
                    print(f"第 {layer} 层检测到梯度冲突，调整聚合策略")

            # 客户端训练
            for client in self.selected_clients:
                client.train_layer_specific(layer, round_idx, self.num_clients, adaptive_alphas[client.id])
            self.receive_models_with_classifier()
            print(f"第 {layer} 层轮次 {round_idx} 完成客户端训练")
            # 检查上传的模型是否有异常
            if hasattr(self, 'fraud_aware_aggregation_enabled') and self.fraud_aware_aggregation_enabled:
                self.fraud_aware_layer_aggregation(layer)
            else:
                self.aggregate_parameters_by_layer(layer)
            if round_idx % self.eval_gap == 0:
                self.current_round = round_idx
                self.evaluate()
            #理论驱动的收敛监控
            metrics = {
                'f1': self.rs_test_f1[-1] if self.rs_test_f1 else 0.0,
                'auc': self.rs_test_auc[-1] if self.rs_test_auc else 0.5,
                'recall': self.rs_test_recall[-1] if self.rs_test_recall else 0.0
            }

            if self.fraud_detection_convergence_monitor(layer, round_idx, metrics):
                print(f"第 {layer} 层在轮次 {round_idx} 早停")
                break

            # 更新客户端性能记录
            if round_idx % self.eval_gap == 0:
                self.update_client_performance_tracking()

            self.Budget.append(time.time() - s_t)
            print(f'第 {layer} 层轮次 {round_idx} 耗时: {self.Budget[-1]:.2f}s')

            # 在服务器的每轮训练结束时调用
            self.log_round_summary(round_idx)

        # 冻结当前层参数（除了最后一层）
        if layer < self.max_layers - 1:
            self.freeze_current_layer_parameters(layer)

    def select_active_clients(self):
        """选择活跃客户端进行训练"""
        active_list = list(self.active_clients_set)
        selected_num = min(self.current_num_join_clients, len(active_list))
        selected_indices = random.sample(active_list, selected_num)
        return [self.clients[i] for i in selected_indices]

    def generate_next_layer_clusters(self, layer_idx):
        """为下一层生成聚类"""
        print(f"开始为第 {layer_idx} 层生成聚类...")

        if layer_idx == 0:
            # 第0层使用全局聚合
            clusters = [[client.id for client in self.selected_clients]]
            print("第 0 层采用全局聚合")
        else:
            # 收集客户端更新
            client_updates = self.collect_client_updates()
            # 使用新的自适应聚类方法
            cluster_ids = self.adaptive_layer_clustering(client_updates, layer_idx)
            clusters = cluster_ids

            print(f"第 {layer_idx} 层生成了 {len(clusters)} 个聚类")
            for i, cluster in enumerate(clusters):
                print(f"聚类 {i}: 客户端 {cluster}")

        return clusters

    def perform_hierarchical_clustering(self, similarity_matrix):
        """基于相似度矩阵进行层次聚类"""
        client_ids = list(similarity_matrix.keys())
        similarity_threshold = 0.7  # 可以根据需要调整

        clusters = []
        assigned = set()

        for client_a in client_ids:
            if client_a in assigned:
                continue

            cluster = [client_a]
            assigned.add(client_a)

            # 找到与当前客户端相似的其他客户端
            for client_b in client_ids:
                if (client_b not in assigned and
                        client_b in similarity_matrix[client_a] and
                        similarity_matrix[client_a][client_b] > similarity_threshold):
                    cluster.append(client_b)
                    assigned.add(client_b)

            clusters.append(cluster)

        return clusters

#######################################################################################################################################
    def compute_client_similarity_matrix(self):
        """基于参数更新模式的相似度计算"""
        active_list = list(self.active_clients_set)
        n = len(active_list)
        similarity_matrix = np.eye(n)

        # 收集参数更新统计信息
        update_signatures = {}
        for client_id in active_list:
            if client_id < len(self.uploaded_models):
                update_signatures[client_id] = self.extract_update_signature(client_id)

        # 计算基于更新模式的相似度
        for i in range(n):
            for j in range(i + 1, n):
                client_i_id = active_list[i]
                client_j_id = active_list[j]

                if client_i_id in update_signatures and client_j_id in update_signatures:
                    sim = self.compute_update_pattern_similarity(
                        update_signatures[client_i_id],
                        update_signatures[client_j_id]
                    )
                    similarity_matrix[i][j] = similarity_matrix[j][i] = sim

        return similarity_matrix

    def compute_client_fraud_rates(self):
        """改进的客户端特征推断"""
        client_characteristics = {}

        for client_id in self.active_clients_set:
            # 修改：训练轮次不足时使用默认值（原版没有此检查）
            if len(self.client_training_history[client_id]) < 15:
                client_characteristics[client_id] = 0.05  # 默认中等风险
                continue

            if client_id < len(self.uploaded_models):
                # 使用更多历史数据（原版用3轮，现在用10轮）
                recent_losses = [record['loss'] for record in self.client_training_history[client_id][-10:]]
                avg_loss = np.mean(recent_losses) if recent_losses else 0.5
                loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else 0.1

                # 调整阈值（原版阈值可能过于敏感）
                if avg_loss < 0.2 and loss_variance < 0.03:  # 原版：avg_loss < 0.3, loss_variance < 0.05
                    client_characteristics[client_id] = 0.02
                elif avg_loss > 0.6 or loss_variance > 0.15:  # 原版：avg_loss > 0.8, loss_variance > 0.2
                    client_characteristics[client_id] = 0.15
                else:
                    client_characteristics[client_id] = 0.05
            else:
                client_characteristics[client_id] = 0.05

        return client_characteristics

    def extract_update_signature(self, client_id):
        """提取客户端的参数更新签名"""
        if client_id >= len(self.uploaded_models):
            return None

        client_model = self.uploaded_models[client_id]
        signature = {}

        # 1. 参数更新幅度分布
        layer_magnitudes = []
        layer_directions = []

        for name, param in client_model.named_parameters():
            if 'weight' in name and param is not None:
                # 计算参数的统计特征
                magnitude = torch.norm(param.data).item()
                sparsity = (param.data.abs() < 1e-6).float().mean().item()

                layer_magnitudes.append(magnitude)
                layer_directions.append(sparsity)

        signature['magnitude_pattern'] = np.array(layer_magnitudes) if layer_magnitudes else np.array([0.0])
        signature['sparsity_pattern'] = np.array(layer_directions) if layer_directions else np.array([0.0])

        # 2. 添加训练损失信息（如果可用）
        if hasattr(self, 'client_loss_history') and client_id in self.client_loss_history:
            recent_losses = self.client_loss_history[client_id][-3:]
            signature['loss_trend'] = np.mean(recent_losses) if recent_losses else 0.0
        else:
            signature['loss_trend'] = 0.0

        return signature

    def compute_update_pattern_similarity(self, sig1, sig2):
        """计算两个更新签名的相似度"""
        if sig1 is None or sig2 is None:
            return 0.0

        # 1. 参数幅度模式相似度
        mag_sim = 1.0 / (1.0 + np.linalg.norm(sig1['magnitude_pattern'] - sig2['magnitude_pattern']))

        # 2. 稀疏性模式相似度
        sparse_sim = 1.0 / (1.0 + np.linalg.norm(sig1['sparsity_pattern'] - sig2['sparsity_pattern']))

        # 3. 损失趋势相似度
        loss_sim = 1.0 / (1.0 + abs(sig1['loss_trend'] - sig2['loss_trend']))

        # 综合相似度
        return 0.4 * mag_sim + 0.4 * sparse_sim + 0.2 * loss_sim

    # def compute_fraud_rate_similarity(self, rate1, rate2):
    #     """计算基于欺诈率的相似度"""
    #     # 使用指数衰减函数计算相似度
    #     diff = abs(rate1 - rate2)
    #     similarity = np.exp(-diff * 10)  # 10是敏感度参数
    #     return similarity




#######################################################################################################################################



    def create_cluster_model(self, cluster_clients, base_layer):
        """为聚类创建模型"""
        # 基于聚类中客户端的模型创建新模型
        cluster_model = copy.deepcopy(self.global_model)
        cluster_model.to(self.device)

        # 收集聚类中客户端的模型参数进行平均
        if len(cluster_clients) > 0:
            client_models = []
            client_weights = []

            for client_id in cluster_clients:
                if client_id < len(self.uploaded_ids) and client_id in self.uploaded_ids:
                    idx = self.uploaded_ids.index(client_id)
                    client_models.append(self.uploaded_models[idx])
                    client_weights.append(self.uploaded_weights[idx])

            # 如果有可用的客户端模型，进行加权平均
            if client_models:
                total_weight = sum(client_weights)
                if total_weight > 0:
                    client_weights = [w / total_weight for w in client_weights]

                    with torch.no_grad():
                        for key in cluster_model.state_dict().keys():
                            if 'bn' not in key:  # 跳过BN层
                                temp = torch.zeros_like(cluster_model.state_dict()[key])
                                for model, weight in zip(client_models, client_weights):
                                    temp += weight * model.state_dict()[key].to(self.device)
                                cluster_model.state_dict()[key].data.copy_(temp)

        return cluster_model

    def fedawa_adaptive_aggregation(self, layer):
        """基于FedAWA理论的自适应层级聚合"""
        if layer not in self.structure:
            return

        for cluster_id, (cluster_clients, cluster_model) in self.structure[layer].items():
            # 收集该聚类中的客户端模型和更新向量
            client_vectors = []
            client_weights = []
            valid_models = []

            for client_id in cluster_clients:
                if client_id in self.uploaded_ids:
                    idx = self.uploaded_ids.index(client_id)
                    client_model = self.uploaded_models[idx]

                    # 【FedAWA核心】计算客户端更新向量
                    client_vector = self.compute_client_update_vector(client_model, cluster_model)
                    if client_vector is not None:
                        client_vectors.append(client_vector)
                        valid_models.append(client_model)
                        client_weights.append(self.uploaded_weights[idx])

            if len(client_vectors) < 2:
                continue

            # 【FedAWA核心】基于向量对齐度计算自适应权重
            adaptive_weights = self.compute_fedawa_weights(client_vectors)

            # 应用自适应权重进行聚合
            with torch.no_grad():
                for key in cluster_model.state_dict().keys():
                    if 'bn' not in key:
                        weighted_param = torch.zeros_like(cluster_model.state_dict()[key])

                        for model, weight in zip(valid_models, adaptive_weights):
                            weighted_param += weight * model.state_dict()[key].to(self.device)

                        cluster_model.state_dict()[key].data.copy_(weighted_param)

    def compute_client_update_vector(self, client_model, reference_model):
        """计算客户端更新向量（FedAWA方法）"""
        try:
            update_vector = []
            with torch.no_grad():
                for (name1, param1), (name2, param2) in zip(
                        client_model.named_parameters(), reference_model.named_parameters()
                ):
                    if 'bn' not in name1:
                        # 计算参数更新差异
                        param_diff = param1.data - param2.data
                        update_vector.append(param_diff.flatten())

            if update_vector:
                return torch.cat(update_vector)
            return None
        except:
            return None

    def compute_fedawa_weights(self, client_vectors):
        """基于FedAWA的对齐度计算自适应权重"""
        n_clients = len(client_vectors)
        alignment_scores = torch.zeros(n_clients)

        # 计算每个客户端与其他客户端的平均对齐度
        for i in range(n_clients):
            total_alignment = 0
            for j in range(n_clients):
                if i != j:
                    # 余弦相似度作为对齐度
                    cosine_sim = F.cosine_similarity(
                        client_vectors[i].unsqueeze(0),
                        client_vectors[j].unsqueeze(0)
                    )
                    total_alignment += torch.clamp(cosine_sim, -1, 1)

            alignment_scores[i] = total_alignment / (n_clients - 1)

        # 【FedAWA核心】将对齐度转换为权重
        # 高对齐度获得更高权重
        raw_weights = torch.softmax(alignment_scores, dim=0)

        # 平滑处理避免权重过于极端
        min_weight = 0.1 / n_clients
        smoothed_weights = raw_weights * (1 - min_weight) + min_weight

        return smoothed_weights.tolist()

    def aggregate_parameters_by_layer(self, layer):
        """按层级聚合参数"""
        if layer == 0:
            # 第0层：全局聚合（原有逻辑）
            self.aggregate_parameters()
        else:
            # 其他层：按聚类聚合
            self.aggregate_parameters_by_clusters(layer)

    def aggregate_parameters_by_clusters(self, layer):
        """改进的按聚类聚合参数 - 使用FedAWA自适应权重"""
        if layer not in self.structure:
            return

        # 使用FedAWA自适应聚合
        self.fedawa_adaptive_aggregation(layer)

        # 记录聚合质量
        self.log_aggregation_quality(layer)

    def log_aggregation_quality(self, layer):
        """记录聚合质量指标"""
        if not hasattr(self, 'aggregation_logs'):
            self.aggregation_logs = []

        # 简单的聚合质量监控
        quality_score = len(self.uploaded_models) / self.num_clients
        self.aggregation_logs.append({
            'layer': layer,
            'quality_score': quality_score,
            'num_clients': len(self.uploaded_models)
        })

    def freeze_current_layer_parameters(self, layer):
        """冻结当前层的参数"""
        if layer in self.structure:
            for cluster_id, (_, cluster_model) in self.structure[layer].items():
                for param in cluster_model.parameters():
                    param.requires_grad = False
            print(f"已冻结第 {layer} 层的参数")

    def update_client_performance_tracking(self):
        """更新客户端性能跟踪"""
        # 这里使用最新的F1分数作为性能指标
        current_f1 = self.rs_test_f1[-1] if self.rs_test_f1 else 0.0

        for client_id in self.active_clients_set:
            if client_id not in self.client_performance_history:
                self.client_performance_history[client_id] = []
            self.client_performance_history[client_id].append(current_f1)

    def update_active_clients_based_on_performance(self):
        """基于性能更新活跃客户端"""
        # 移除持续表现差的客户端
        if len(self.client_performance_history) > 0:
            avg_performance = np.mean([np.mean(scores[-3:]) if len(scores) >= 3 else np.mean(scores)
                                       for scores in self.client_performance_history.values()])

            clients_to_remove = []
            for client_id in self.active_clients_set:
                if client_id in self.client_performance_history:
                    client_scores = self.client_performance_history[client_id]
                    recent_avg = np.mean(client_scores[-3:]) if len(client_scores) >= 3 else np.mean(client_scores)

                    # 如果客户端表现持续低于平均水平的70%，则移除
                    if recent_avg < 0.7 * avg_performance and len(client_scores) >= 3:
                        clients_to_remove.append(client_id)

            for client_id in clients_to_remove:
                self.active_clients_set.discard(client_id)

            if clients_to_remove:
                print(f"移除表现不佳的客户端: {clients_to_remove}")



    def send_models_with_classifier(self, layer):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            # 只发送当前层的聚合结构
            client.set_parameters_with_classifier(
                self.structure,  # 只发送结构信息
                self.classifier_list,
                layer  # 传递当前层级
            )
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)



    def receive_models_with_classifier(self):
        assert (len(self.selected_clients) > 0)   #使用assert确保可挑选的客户端始终大于0

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.classifier_list = {}
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(copy.deepcopy(client.model))
                self.classifier_list[client.id] = copy.deepcopy(client.offline_model.classifier)
                received_model = self.uploaded_models[-1]


        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        for client in active_clients:
            client_id = client.id

            # 收集模型参数
            if hasattr(client, 'model'):
                params_dict = {}
                for name, param in client.model.named_parameters():
                    params_dict[name] = param.data.clone().detach()
                self.client_parameters_history[client_id].append(params_dict)
                if len(self.client_parameters_history[client_id]) > 10:
                    self.client_parameters_history[client_id].pop(0)

            # 收集训练历史
            if hasattr(client, 'current_train_loss'):
                training_record = {
                    'round': getattr(self, 'current_round', 0),
                    'loss': client.current_train_loss,
                    'client_id': client_id
                }
                self.client_training_history[client_id].append(training_record)
                if len(self.client_training_history[client_id]) > 20:
                    self.client_training_history[client_id].pop(0)

    def aggregate_parameters(self):
        """改进的参数聚合 - 使用梯度裁剪防止爆炸"""
        assert (len(self.uploaded_models) > 0)

        # 标准FedAvg聚合
        new_global_params = self.fedavg_aggregate()

        # PPO风格的参数裁剪（如果启用）
        if self.ppo_enabled and self.old_global_params is not None:
            new_global_params = self.ppo_clip_parameters(new_global_params)

        # 更新全局模型
        self.update_global_model(new_global_params)

        # 保存历史参数
        self.old_global_params = copy.deepcopy(new_global_params)

    def fedavg_aggregate(self):
        """改进的FedAvg聚合 - 添加数值稳定性检查"""
        aggregated_params = {}
        first_model = self.uploaded_models[0]

        for key in first_model.state_dict().keys():
            if 'bn' not in key and 'num_batches_tracked' not in key:
                temp = torch.zeros_like(first_model.state_dict()[key], dtype=torch.float32)

                for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                    param_contribution = w * client_model.state_dict()[key].to(self.device)
                    # 检查数值稳定性
                    if torch.isnan(param_contribution).any() or torch.isinf(param_contribution).any():
                        print(f"Warning: NaN/Inf detected in parameter {key}, skipping this client")
                        continue
                    temp += param_contribution

                aggregated_params[key] = temp.clone()

        return aggregated_params

    def ppo_clip_parameters(self, new_params):
        """改进的PPO裁剪 - 降低裁剪强度"""
        clipped_params = {}
        # 降低裁剪范围
        epsilon = max(0.05, self.ppo_epsilon * 0.5)  # 原来的一半

        for key in new_params.keys():
            if key in self.old_global_params:
                old_param = self.old_global_params[key]
                new_param = new_params[key]

                # 计算相对变化
                param_diff = new_param - old_param
                param_ratio = param_diff / (torch.abs(old_param) + 1e-8)

                # 温和的裁剪
                clipped_ratio = torch.clamp(param_ratio, -epsilon, epsilon)
                clipped_params[key] = old_param + clipped_ratio * torch.abs(old_param)
            else:
                clipped_params[key] = new_params[key]

        return clipped_params

    def update_global_model(self, new_params):
        """更新全局模型参数"""
        with torch.no_grad():
            for key, param in new_params.items():
                if key in self.global_model.state_dict():
                    self.global_model.state_dict()[key].data.copy_(param)

    def evaluate(self,acc=None, loss=None):
        """
                在服务器端使用全局测试集评估全局模型。
                """
        if self.global_test_data is None:
            print("无法进行全局评估：全局测试集未加载。")
            return {}

        self.global_model.eval()
        self.global_model.to(self.device)
        all_preds_prob = []
        all_labels = []

        test_loader = DataLoader(
            self.global_test_data,
            batch_size=self.args.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.global_model(x)

                # 存储预测概率和真实标签
                all_preds_prob.extend(outputs.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

        all_preds_prob = np.array(all_preds_prob)
        all_labels = np.array(all_labels)

        # 计算评估指标
        binary_preds = (all_preds_prob > 0.5).astype(int)

        accuracy = np.mean(binary_preds == all_labels)
        f1 = f1_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds_prob)
        auprc = average_precision_score(all_labels, all_preds_prob)
        cm = confusion_matrix(all_labels, binary_preds)


        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("-------------------------------------------\n")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'auprc': auprc
        }




    def whether_early_stop(self,score):
        if self.best_score is None:
            self.best_score = score
            return
        if self.early_stop_mode == 'max':
            if score - self.best_score > self.min_delta:
                old = self.best_score
                self.best_score = score
                self.num_bad_rounds = 0
                print(f"Early stop monitor: improved metric from {old:.4f} to {score:.4f}")
            else:
                self.num_bad_rounds += 1
                print(f"Early stop monitor: no improvement for {self.num_bad_rounds} rounds")
        else:
            if self.best_score - score > self.min_delta:
                old = self.best_score
                self.best_score = score
                self.num_bad_rounds = 0
                print(f"Early stop monitor: improved metric from {old:.4f} to {score:.4f}")
            else:
                self.num_bad_rounds += 1
                print(f"Early stop monitor: no improvement for {self.num_bad_rounds} rounds")

        if self.num_bad_rounds >= self.patience:
            self.early_stop = True
            print(f"Early stopping triggered after {self.num_bad_rounds} rounds without improvement")

    # def init_layer(self):
    #     self.structure[0][0] = [np.arange(self.num_clients),]

    def log_round_summary(self, round_num):
        """每轮结束的汇总信息"""
        total_anomalies = sum(
            1 for client in self.clients if hasattr(client, 'anomaly_count') and client.anomaly_count > 0)

        if total_anomalies > 0:
            print(f"⚠️  Round {round_num}: {total_anomalies}/{len(self.clients)} clients showed anomalies")

            # 列出异常最多的客户端
            anomaly_clients = [(i, client.anomaly_count) for i, client in enumerate(self.clients)
                               if hasattr(client, 'anomaly_count') and client.anomaly_count > 2]
            if anomaly_clients:
                print(f"   High-anomaly clients: {[f'C{i}({count})' for i, count in anomaly_clients]}")

        # 重置异常计数器，避免累积
        for client in self.clients:
            if hasattr(client, 'anomaly_count'):
                client.anomaly_count = max(0, client.anomaly_count - 1)  # 逐渐减少计数

    def plot_metrics(self):
        """绘制所有评估指标的折线图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Evaluation Metrics Over Training Rounds', fontsize=16)

            # 绘制准确率
            axes[0, 0].plot(self.eval_rounds, self.rs_total_test_acc, 'b-', label='Total Test Accuracy')
            axes[0, 0].plot(self.eval_rounds, self.rs_online_test_acc, 'g--', label='Online Test Accuracy')
            axes[0, 0].plot(self.eval_rounds, self.rs_offline_test_acc, 'r-.', label='Offline Test Accuracy')
            axes[0, 0].set_xlabel('Training Rounds')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Over Rounds')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 绘制Recall、F1-score和AUPRC
            axes[0, 1].plot(self.eval_rounds, self.rs_test_recall, 'b-', label='Recall')
            axes[0, 1].plot(self.eval_rounds, self.rs_test_f1, 'g--', label='F1-score')
            axes[0, 1].plot(self.eval_rounds, self.rs_test_auprc, 'r-.', label='AUPRC')
            axes[0, 1].set_xlabel('Training Rounds')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Recall, F1-score and AUPRC Over Rounds')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 绘制AUC
            axes[1, 0].plot(self.eval_rounds, self.rs_test_auc, 'b-', label='ROC-AUC')
            axes[1, 0].set_xlabel('Training Rounds')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_title('ROC-AUC Over Rounds')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # 绘制损失
            axes[1, 1].plot(self.eval_rounds, self.rs_total_train_loss, 'b-', label='Total Train Loss')
            axes[1, 1].plot(self.eval_rounds, self.rs_online_train_loss, 'g--', label='Online Train Loss')
            axes[1, 1].plot(self.eval_rounds, self.rs_offline_train_loss, 'r-.', label='Offline Train Loss')
            axes[1, 1].set_xlabel('Training Rounds')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss Over Rounds')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            plt.tight_layout()

            # 保存图表
            plt.savefig('C:/Users/Study/PycharmProjects/swu-fl/results/metrics_trend.png', dpi=300, bbox_inches='tight')
            plt.savefig('C:/Users/Study/PycharmProjects/swu-fl/results/metrics_trend.pdf', bbox_inches='tight')
            plt.close()

            print("Metrics plots saved to C:/Users/Study/PycharmProjects/swu-fl/results/metrics_trend.png and C:/Users/Study/PycharmProjects/swu-fl/results/metrics_trend.pdf")

        except ImportError:
            print("Matplotlib or Seaborn not installed. Cannot generate plots.")
            print("Please install them with: pip install matplotlib seaborn")

    def detect_fraud_gradient_conflicts(self, layer):
        """
        原先可能基于客户端的精确欺诈比例来检测梯度冲突。
        现在我们只使用离散等级信息，并结合梯度/更新签名来做判定。

        如果等级差 >= 2（即 0 vs 2），则直接标为高冲突概率；
        如果等级差 = 1，则进一步检查更新签名/梯度范数差异以决定。
        """
        active_list = sorted(list(self.active_clients_set))
        # 若客户端过少，返回 False
        if len(active_list) < 2:
            return False

        # 读取 fraud level（整数）
        levels = {}
        for client in [self.clients[i] for i in active_list]:
            cid = client.id
            levels[cid] = self.get_client_fraud_level(client)

        # 快速判定：若存在明显等级差异的客户端对，则认为冲突
        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                a = active_list[i]
                b = active_list[j]
                la = levels.get(a, 1)
                lb = levels.get(b, 1)
                if abs(la - lb) >= 2:
                    # 0 vs 2：强不一致信号
                    return True
                elif abs(la - lb) == 1:
                    # 1级差需要进一步检查参数更新签名或梯度范数
                    sig_a = self.extract_update_signature(a)
                    sig_b = self.extract_update_signature(b)
                    # 如果签名不可用，则保守地认为冲突
                    if sig_a is None or sig_b is None:
                        return True
                    # 计算 magnitude pattern 差距
                    mag_diff = np.linalg.norm(
                        sig_a.get('magnitude_pattern', np.zeros(1)) - sig_b.get('magnitude_pattern', np.zeros(1)))
                    # 若差距较大则认为冲突（阈值可调）
                    if mag_diff > 1.0:
                        return True
        return False

    # -----------------
    # 辅助方法：放到同一个类里
    # -----------------
    def _compute_client_update_vector_for_conflict(self, client_id, layer, use_last_layer_only=True):
        """
        尝试从 self.uploaded_models 或 client_parameters_history 获取客户端的“更新向量”。
        返回 np.array(flattened vector) 或 None。
        优先使用最新上传的 model 差值（与上一轮 global model 的差）。
        如果只需最后一层/分类器，提取对应参数以减少噪声。
        """
        # 优先使用 self.uploaded_models（如果存在）
        # uploaded_models 是训练后客户端上传的 model 实例列表（server 在聚合前有这个）
        if hasattr(self, 'uploaded_models') and len(self.uploaded_models) > 0:
            # uploaded_ids 对应每个 model 的 client id 顺序
            for model, cid in zip(self.uploaded_models, self.uploaded_ids):
                if cid == client_id:
                    try:
                        # 计算 model.parameters() - self.global_model.parameters()
                        global_state = self.global_model.state_dict()
                        vecs = []
                        for name, param in model.state_dict().items():
                            # 可选：只考虑 classifier 层（名字中含 'classifier' 或最后一个 linear）
                            if use_last_layer_only:
                                if 'classifier' not in name and 'fc' not in name and 'head' not in name:
                                    continue
                            # skip bn tracking buffers
                            if 'num_batches_tracked' in name:
                                continue
                            p = param.float().cpu().numpy()
                            g = p - global_state[name].float().cpu().numpy()
                            vecs.append(g.ravel())
                        if not vecs:
                            return None
                        flat = np.concatenate(vecs)
                        return flat
                    except Exception as e:
                        return None

        # 否则尝试从 client_parameters_history 中构建（history 中应包含 state_dict 或差分）
        if hasattr(self, 'client_parameters_history') and client_id < len(self.client_parameters_history):
            history = self.client_parameters_history[client_id]
            if len(history) >= 1:
                # 取最近一次上传的 state_dict（假设 history 存的是 state_dict）
                last = history[-1]
                if isinstance(last, dict):
                    vecs = []
                    for name, v in last.items():
                        if use_last_layer_only:
                            if 'classifier' not in name and 'fc' not in name and 'head' not in name:
                                continue
                        try:
                            arr = v.cpu().numpy()
                            vecs.append(arr.ravel())
                        except:
                            continue
                    if vecs:
                        return np.concatenate(vecs)
        return None

    def _cosine_similarity(self, a, b, eps=1e-8):
        """a, b are numpy arrays"""
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb + eps))

    def compute_fraud_aware_client_vector(self, client):
        """
        生成一个“欺诈感知向量”，用于后续聚合/加权/聚类。
        原来可能使用精确比例构建，现在使用 one-hot 的等级向量 + 可选的本地训练损失均值
        以丰富特征，但**绝不包含精确欺诈数量**。
        返回 numpy 向量（例如长度 4: [one-hot(3), avg_loss]）
        """
        level = self.get_client_fraud_level(client)
        one_hot = np.zeros(3, dtype=float)
        if level in (0, 1, 2):
            one_hot[level] = 1.0
        else:
            one_hot[1] = 1.0

        # 从训练历史中取平均训练损失作为额外特征（不含标签信息）
        cid = client.id
        loss_feat = 0.0
        if cid < len(self.client_training_history) and len(self.client_training_history[cid]) > 0:
            recent = [r.get('loss', 0.0) for r in self.client_training_history[cid][-5:]]
            loss_feat = float(np.mean(recent))
        else:
            loss_feat = 0.5

        vec = np.concatenate([one_hot, np.array([loss_feat])])
        return vec



    def compute_vector_similarity(self, vec1, vec2):
        """计算两个客户端向量的相似度"""
        # 简化的相似度计算，基于关键特征
        if vec1['type'] != vec2['type']:
            return 0.1  # 类型不同，相似度很低

        fraud_ratio_diff = abs(vec1.get('fraud_ratio', 0) - vec2.get('fraud_ratio', 0))
        similarity = max(0, 1.0 - fraud_ratio_diff * 5)  # 欺诈率差异越大，相似度越低

        return similarity

    def fraud_aware_layer_aggregation(self, layer):
        """基于欺诈检测特点的层级聚合"""
        if not hasattr(self, 'uploaded_models') or len(self.uploaded_models) == 0:
            return

        # 为每个上传模型的客户端计算欺诈感知权重
        ordered_uploaded_models = []
        aggregation_weights = []
        for client_id in self.uploaded_ids:
            client = self.clients[client_id]
            weight = self.compute_fraud_aware_weights(client)
            aggregation_weights.append(weight)

            # 使用索引来保证顺序
            model_index = self.uploaded_ids.index(client_id)
            ordered_uploaded_models.append(self.uploaded_models[model_index])

        total_weight = sum(aggregation_weights)
        if total_weight > 0:
            aggregation_weights = [w / total_weight for w in aggregation_weights]

        # 执行加权聚合
        # self.weighted_parameter_aggregation(ordered_uploaded_models, aggregation_weights)
        aggregated_params = self.weighted_parameter_aggregation(ordered_uploaded_models, aggregation_weights)

        # 将聚合参数应用到全局模型
        if aggregated_params:
            self.update_global_model(aggregated_params)

        print(f"第 {layer} 层使用欺诈感知聚合，权重分布: {[f'{w:.3f}' for w in aggregation_weights[:5]]}")



    def compute_fraud_aware_weights(self, client):
        """
        为聚合或为某些策略分配权重。不得使用精确样本数。
        使用等级映射到权重的策略（可自定义映射或学习得到）。
        返回单个浮点数权重。
        """
        level = self.get_client_fraud_level(client)
        # 简单策略（可调整或用函数映射）
        # 低欺诈 -> 普通权重
        # 中等 -> 轻微增加（因为中等样本可能更具信息性）
        # 高 -> 适当增加（因为高欺诈样本对模型影响力重要）
        if level == 0:
            return 1.0
        elif level == 1:
            return 1.2
        elif level == 2:
            return 1.5
        else:
            return 1.0


    def weighted_parameter_aggregation(self, models, weights, trim_ratio=0.2):
        """
        Perform parameter aggregation with:
          - weight clipping & normalization (caller should pass normalized weights),
          - if evidence of suspicious clients (e.g., max(weight) too large OR detected conflict),
            do coordinate-wise trimmed mean as robust fallback.
        """
        if len(models) == 0:
            return
        # normalize defensively
        w = np.array(weights, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        dominant = (w.max() > 0.45)
        conflict_flag = getattr(self, 'conflict_mode_active', False)
        use_trim = dominant or conflict_flag

        aggregated_params = {}
        first_model_dict = models[0].state_dict()
        # global_dict = self.global_model.state_dict()
        for key in first_model_dict.keys():
            if 'num_batches_tracked' in key:
                continue

            parts = []
            for model in models:
                sd = model.state_dict()
                if key in sd:
                    parts.append(sd[key].detach().cpu())
            if len(parts) == 0:
                continue

            stacked = torch.stack(parts, dim=0)

            if use_trim and stacked.shape[0] >= 3:
                m = stacked.shape[0]
                k = int(max(1, math.floor(trim_ratio * m)))
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[k:m - k].mean(dim=0)
                agg = trimmed.to(self.device)
            else:
                ws = torch.tensor(w, dtype=torch.float32).view(-1, *([1] * (stacked.dim() - 1)))
                agg = (stacked * ws).sum(dim=0).to(self.device)

            # 核心修改：将聚合结果存入新字典
            aggregated_params[key] = agg

        return aggregated_params  # 返回聚合后的参数

    def fraud_detection_convergence_monitor(self, layer, round_idx, metrics, patience=3, min_delta=1e-4):
        # metrics: dict with 'f1','auc','recall'
        if not hasattr(self, 'layer_convergence_history'):
            self.layer_convergence_history = {}
        if layer not in self.layer_convergence_history:
            self.layer_convergence_history[layer] = {'f1': [], 'auc': [], 'recall': [], 'bad_rounds': 0,
                                                     'best_f1': -1.0, 'best_auc': -1.0}
        hist = self.layer_convergence_history[layer]
        f1 = metrics.get('f1', 0.0)
        auc = metrics.get('auc', 0.5)
        recall = metrics.get('recall', 0.0)
        hist['f1'].append(f1);
        hist['auc'].append(auc);
        hist['recall'].append(recall)

        improved = False
        # strict improvement on F1
        if f1 > hist['best_f1'] + min_delta:
            hist['best_f1'] = f1
            improved = True
        # if F1 not improved but AUC improved significantly, allow as improvement
        elif auc > hist['best_auc'] + min_delta:
            hist['best_auc'] = auc
            improved = True

        if improved:
            hist['bad_rounds'] = 0
            return False
        else:
            hist['bad_rounds'] += 1
            # but if recall improves, be lenient (don't increment bad_rounds or decrement threshold)
            if recall > (max(hist['recall']) if hist['recall'][:-1] else 0.0):
                hist['bad_rounds'] = max(0, hist['bad_rounds'] - 1)
            return hist['bad_rounds'] >= patience

    def adjust_aggregation_strategy(self, layer, mode):
        """根据检测到的问题调整聚合策略"""
        if mode == "conflict_mode":
            # 启用欺诈感知聚合
            self.fraud_aware_aggregation_enabled = True
            # 可以进一步调整其他参数
        else:
            self.fraud_aware_aggregation_enabled = False

    def get_client_fraud_level(self, client):
        """
        服务器端调用此接口以获取客户端的欺诈等级（整数），
        而不是精确的欺诈计数或比例。

        client: 客户端对象（clientMyMethod 实例）
        返回值: integer in {0,1,2}（如果失败则返回 1 作为中性值）
        """
        try:
            if hasattr(client, 'get_client_fraud_level'):
                level = client.get_client_fraud_level()
                # 防御：确保返回值受控
                if level is None:
                    return 1
                if isinstance(level, int) and level in (0, 1, 2):
                    return int(level)
                else:
                    # 如果客户端返回非预期值，退化为中间等级
                    return 1
            else:
                return 1
        except Exception:
            # 保守策略：异常时返回中等风险等级
            return 1

#--------------------------------------------------------------------------------------------------------------------------------------------------
    def compute_gradient_similarity(self, client_updates):
        """计算客户端之间的梯度相似度"""
        similarities = {}
        client_ids = list(client_updates.keys())

        for i, client_a in enumerate(client_ids):
            similarities[client_a] = {}
            grad_a = self.flatten_gradients(client_updates[client_a])

            for j, client_b in enumerate(client_ids):
                if i <= j:
                    grad_b = self.flatten_gradients(client_updates[client_b])
                    # 使用余弦相似度
                    similarity = torch.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0))
                    similarities[client_a][client_b] = similarity.item()
                    if client_b not in similarities:
                        similarities[client_b] = {}
                    similarities[client_b][client_a] = similarity.item()

        return similarities

    def flatten_gradients(self, model_params):
        """将模型参数展平为一维向量"""
        flattened = []
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor) and param.requires_grad:
                flattened.append(param.flatten())

        if not flattened:
            # 如果没有可训练参数，返回零向量
            return torch.zeros(1, device=self.device)

        return torch.cat(flattened)

    def adaptive_layer_clustering(self, client_updates, layer_idx):
        """自适应层级聚类"""
        similarities = self.compute_gradient_similarity(client_updates)

        # 根据层级调整相似度阈值
        similarity_threshold = 0.8 - 0.1 * layer_idx  # 随层级递减

        client_ids = list(client_updates.keys())
        clusters = []
        assigned = set()

        for client_a in client_ids:
            if client_a in assigned:
                continue

            cluster = [client_a]
            assigned.add(client_a)

            for client_b in client_ids:
                if (client_b not in assigned and
                        client_b in similarities[client_a] and
                        similarities[client_a][client_b] > similarity_threshold):
                    cluster.append(client_b)
                    assigned.add(client_b)

            clusters.append(cluster)

        return clusters

    def progressive_layer_training(self, layer_idx):
        """渐进式层级训练，支持参数解冻"""
        if layer_idx > 0:
            # 部分解冻前一层参数，允许微调
            prev_layer_params = f"classifier.{2 * (layer_idx - 1)}"  # 对应分类器层
            for name, param in self.global_model.named_parameters():
                if prev_layer_params in name:
                    param.requires_grad = True  # 解冻允许微调
                    # 使用更小的学习率
                    param.lr_scale = 0.1

        # 当前层正常训练
        current_layer_params = f"classifier.{2 * layer_idx}"
        for name, param in self.global_model.named_parameters():
            if current_layer_params in name:
                param.requires_grad = True
                param.lr_scale = 1.0

    def collect_client_updates(self):
        """收集客户端模型更新用于聚类分析"""
        updates = {}
        for client in self.selected_clients:
            updates[client.id] = {}
            for name, param in client.model.named_parameters():
                if param.requires_grad:
                    updates[client.id][name] = param.data.clone()
        return updates

    def train_cluster_clients(self, cluster_clients, layer_idx, round_idx,alpha):
        """训练聚类内的客户端"""
        for client in cluster_clients:
            # 设置自适应学习率
            adaptive_lr = client.get_adaptive_lr()
            for param_group in client.optimizer.param_groups:
                param_group['lr'] = adaptive_lr
            print(f"客户端 {client.id}: 自适应学习率 {adaptive_lr:.6f}")

            client_alpha = alpha[client.id]
            # 训练客户端
            client.train_layer_specific(layer_idx, round_idx, len(cluster_clients), client_alpha)

            # 更新学习率历史信息
            training_loss = getattr(client, 'training_loss', 1.0)
            global_loss = getattr(self, 'global_loss', 1.0)
            client.update_lr_history(training_loss, global_loss, round_idx)

    def aggregate_cluster(self, models, clients):
        """聚合聚类内的模型更新"""
        # 使用FedAvg方式聚合
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            if any(key in model.state_dict() for model in models):
                global_dict[key] = torch.stack([
                    model.state_dict()[key] * (len(client.train_samples) /
                                               sum(len(c.train_samples) for c in clients))
                    for model, client in zip(models, clients)
                    if key in model.state_dict()
                ]).sum(0)

        self.global_model.load_state_dict(global_dict)

    def compute_adaptive_alpha(self, noisy_client_ratios):
        """
        根据客户端报告的带噪比例，计算每个客户端的自适应alpha值。
        """
        alphas = {}
        min_alpha = 0.2  # 设置蒸馏权重的下限
        max_alpha = 0.8  # 设置蒸馏权重的上限

        for client_id, noisy_ratio in noisy_client_ratios.items():

            alpha = noisy_ratio

            # 将 alpha 限制在预设范围内，以保证训练稳定性
            alphas[client_id] = max(min_alpha, min(max_alpha, alpha))

        return alphas

    def get_noisy_pos_ratios(self, clients): # 接收客户端对象列表
        """
        向指定的客户端请求其带噪的正样本比例。
        """
        noisy_ratios = {}
        # clients 是一个包含客户端对象的列表，而不是 client_id 列表
        for client in clients:
            # 直接调用客户端对象的 get_noisy_pos_ratio 方法
            noisy_ratios[client.id] = client.get_noisy_pos_ratio()
        return noisy_ratios

# --------------------------------------------------------------------------------------------------------------------------------------------------


