import copy
import time
import random
import torch
import numpy as np

from system.flcore.clients.clientavg import clientAVG
from system.flcore.servers.serverbase import Server
from system.flcore.servers.serveravg import FedAvg
from system.flcore.clients.clientMyMethod import clientMyMethod
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, auc,  roc_auc_score
import torch.nn.functional as F



class MyMethod(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)

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
        self.patience = 3  # 更短的耐心，因为欺诈检测收敛慢

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
        """训练当前层级"""
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
            self.send_models_with_classifier()

            if round_idx % self.eval_gap == 0:
                print(f"\n--------- 第 {layer} 层，轮次 {round_idx} ---------")
                self.current_round = round_idx # 为了区分不同层的轮次
                self.evaluate()
            #梯度冲突检测
            if round_idx > 0:
                conflict_detected = self.detect_fraud_gradient_conflicts(layer)
                if conflict_detected:
                    self.adjust_aggregation_strategy(layer, "conflict_mode")
                    print(f"第 {layer} 层检测到梯度冲突，调整聚合策略")

            # 客户端训练
            for client in self.selected_clients:
                client.train_layer_specific(layer, round_idx, self.num_clients)
            print(f"第 {layer} 层轮次 {round_idx} 完成客户端训练")
            # 检查上传的模型是否有异常
            if hasattr(self, 'fraud_aware_aggregation_enabled') and self.fraud_aware_aggregation_enabled:
                self.fraud_aware_layer_aggregation(layer)
            else:
                self.aggregate_parameters_by_layer(layer)

            # 【新增】理论驱动的收敛监控
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

    def generate_next_layer_clusters(self, current_layer):
        """生成下一层的聚类"""
        next_layer = current_layer + 1
        self.structure[next_layer] = {}

        print(f"\n开始为第 {next_layer} 层生成聚类...")

        # 计算客户端相似度矩阵
        similarity_matrix = self.compute_client_similarity_matrix()

        # 执行层次聚类
        clusters = self.perform_hierarchical_clustering(similarity_matrix)

        # 为每个聚类创建模型
        cluster_id = 0
        for cluster_clients in clusters:
            if len(cluster_clients) >= self.min_clients_per_cluster:
                cluster_model = self.create_cluster_model(cluster_clients, current_layer)
                self.structure[next_layer][cluster_id] = [np.array(cluster_clients), cluster_model]
                print(f"聚类 {cluster_id}: 客户端 {cluster_clients}")
                cluster_id += 1

        print(f"第 {next_layer} 层生成了 {len(self.structure[next_layer])} 个聚类")

        # 更新活跃客户端（移除表现差的客户端）
        self.update_active_clients_based_on_performance()
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
    def compute_fraud_rate_similarity(self, rate1, rate2):
        """计算基于欺诈率的相似度"""
        # 使用指数衰减函数计算相似度
        diff = abs(rate1 - rate2)
        similarity = np.exp(-diff * 10)  # 10是敏感度参数
        return similarity



    def cluster_by_similarity(self, client_list, similarity_matrix, active_list, target_size=3):
        """基于相似度对客户端进行聚类"""
        if len(client_list) <= target_size:
            return [client_list]

        clusters = []
        remaining = client_list.copy()

        while len(remaining) > 0:
            if len(remaining) <= target_size:
                clusters.append(remaining)
                break

            # 选择第一个客户端作为种子
            seed = remaining.pop(0)
            current_cluster = [seed]

            # 找到最相似的客户端加入当前聚类
            while len(current_cluster) < target_size and len(remaining) > 0:
                best_client = None
                best_similarity = -1

                for candidate in remaining:
                    # 计算候选客户端与当前聚类的平均相似度
                    avg_sim = 0
                    for cluster_member in current_cluster:
                        i = active_list.index(cluster_member)
                        j = active_list.index(candidate)
                        avg_sim += similarity_matrix[i][j]
                    avg_sim /= len(current_cluster)

                    if avg_sim > best_similarity:
                        best_similarity = avg_sim
                        best_client = candidate

                if best_client:
                    current_cluster.append(best_client)
                    remaining.remove(best_client)
                else:
                    break

            clusters.append(current_cluster)

        return clusters
#######################################################################################################################################
    def compute_parameter_similarity(self, client_i_id, client_j_id):
        """基于模型参数计算相似度"""
        if (client_i_id >= len(self.client_parameters_history) or
                client_j_id >= len(self.client_parameters_history) or
                len(self.client_parameters_history[client_i_id]) == 0 or
                len(self.client_parameters_history[client_j_id]) == 0):
            return 0.5  # 默认中等相似度

        params_i = self.client_parameters_history[client_i_id][-1]
        params_j = self.client_parameters_history[client_j_id][-1]

        similarity = 0.0
        total_params = 0

        for key in params_i.keys():
            if key in params_j and 'bn' not in key:
                param_i_flat = params_i[key].flatten()
                param_j_flat = params_j[key].flatten()

                if param_i_flat.shape == param_j_flat.shape:
                    cos_sim = F.cosine_similarity(param_i_flat.unsqueeze(0),
                                                  param_j_flat.unsqueeze(0))
                    similarity += cos_sim.item() * param_i_flat.numel()
                    total_params += param_i_flat.numel()

        return similarity / total_params if total_params > 0 else 0.5


    def perform_hierarchical_clustering(self, similarity_matrix):
        """改进的金融场景层次聚类"""
        active_list = list(self.active_clients_set)
        fraud_rates = self.compute_client_fraud_rates()

        # 按欺诈率分组
        high_fraud_clients = []  # > 0.1 (10%)
        medium_fraud_clients = []  # 0.02-0.1 (2%-10%)
        low_fraud_clients = []  # < 0.02 (2%)

        for client_id in active_list:
            fraud_rate = fraud_rates[client_id]
            if fraud_rate > 0.1:
                high_fraud_clients.append(client_id)
            elif fraud_rate > 0.02:
                medium_fraud_clients.append(client_id)
            else:
                low_fraud_clients.append(client_id)

        clusters = []

        # 高欺诈率客户端：单独或两两配对
        for i, client_id in enumerate(high_fraud_clients):
            if i % 2 == 0 and i + 1 < len(high_fraud_clients):
                clusters.append([high_fraud_clients[i], high_fraud_clients[i + 1]])
            elif i == len(high_fraud_clients) - 1:
                clusters.append([client_id])

        # 中等欺诈率：3-4个一组
        if len(medium_fraud_clients) > 0:
            # 使用相似度进一步细分
            medium_clusters = self.cluster_by_similarity(
                medium_fraud_clients, similarity_matrix, active_list, target_size=3
            )
            clusters.extend(medium_clusters)

        # 低欺诈率：可以组成较大cluster
        if len(low_fraud_clients) > 0:
            if len(low_fraud_clients) <= 6:
                clusters.append(low_fraud_clients)
            else:
                # 分成两组
                mid = len(low_fraud_clients) // 2
                clusters.append(low_fraud_clients[:mid])
                clusters.append(low_fraud_clients[mid:])

        print(
            f"欺诈率分布 - 高风险: {len(high_fraud_clients)}, 中风险: {len(medium_fraud_clients)}, 低风险: {len(low_fraud_clients)}")
        return clusters

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
#####################################################################################################################################################
    #原版train!!!!!!!!!!!!!!!!!!!!
    # def train(self):
    #
    #     self.structure[0][0] = [np.arange(self.num_clients), copy.deepcopy(self.global_model)]
    #     for i in range(self.global_rounds+1):
    #         s_t = time.time()     #记录本轮训练的时间消耗
    #         self.selected_clients = self.select_clients()    #选择参与本轮训练的客户端
    #         self.send_models_with_classifier()    #将全局模型发送给选中的客户端
    #
    #         if i%self.eval_gap == 0:         #eval_gap应该是评估间隔，如果轮数能被评估间隔整除，就进行一次模型评估
    #             print(f"\n-------------Round number: {i}-------------")
    #             print("\nEvaluate global model")
    #             self.current_round = i
    #             self.evaluate()
    #
    #         for client in self.selected_clients:
    #             client.train(i,self.num_clients)
    #
    #         # threads = [Thread(target=client.train)
    #         #            for client in self.selected_clients]
    #         # [t.start() for t in threads]
    #         # [t.join() for t in threads]
    #
    #         self.receive_models_with_classifier()        #从客户端接收模型参数
    #         if self.dlg_eval and i%self.dlg_gap == 0:
    #             self.call_dlg(i)
    #         self.aggregate_parameters()
    #         if self.early_stop:
    #             print("Early Stopping Triggered at Round ",i)
    #             break
    #
    #         self.Budget.append(time.time() - s_t)
    #         print('-'*25, 'time cost', '-'*25, self.Budget[-1])
    #
    #         if self.auto_break and self.check_done(acc_lss=[self.rs_total_test_acc], top_cnt=self.top_cnt):
    #             break
    #
    #     print("\nBest accuracy.")
    #     # self.print_(max(self.rs_test_acc), max(
    #     #     self.rs_train_acc), min(self.rs_train_loss))
    #     print(max(self.rs_total_test_acc))
    #     print("\nAverage time cost per round.")
    #     print(sum(self.Budget[1:])/len(self.Budget[1:]))
    #
    #
    #     for value in self.rs_total_test_acc:
    #         self.rs_test_acc.append(value)
    #     for value in self.rs_total_train_loss:
    #         self.rs_train_loss.append(value)
    #
    #     self.save_results()
    #     self.save_global_model()
    #
    #     self.plot_metrics()
    #
    #     if self.num_new_clients > 0:
    #         self.eval_new_clients = True
    #         self.set_new_clients(clientAVG)
    #         print(f"\n-------------Fine tuning round-------------")
    #         print("\nEvaluate new clients")
    #         self.evaluate()
#####################################################################################################################################################
    def send_models_with_classifier(self):
        assert (len(self.clients) > 0)  # 用assrt来确保客户端列表的客户端至少有一个，如果len(self.clients) > 0不成立那么程序就会终止运行并报错

        for client in self.clients:
            start_time = time.time()  # 设置开始时间，用于后面用结束时间减去开始时间来计算发送，接受模型用时

            client.set_parameters_with_classifier(self.global_model,self.classifier_list,self.structure)  # 将全局模型的参数发送给客户端

            client.send_time_cost['num_rounds'] += 1  # 表示发送模型的伦次数+1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)  # 计算客户端发送并接受模型参数用时



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
                self.uploaded_models.append(client.model)
                self.classifier_list[client.id] = copy.deepcopy(client.offline_model.classifier)
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
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        ids, num_samples, online_correct, offline_correct, total_correct, tot_auc, all_y_true, all_y_prob = stats
        online_test_acc = sum(online_correct) / sum(num_samples)
        offline_test_acc = sum(offline_correct) / sum(num_samples)
        total_test_acc = sum(total_correct) / sum(num_samples)


        # test_auc = sum(tot_auc) / sum(num_samples)

        online_train_loss = sum(stats_train[2]) / sum(stats_train[1])
        offline_train_loss = sum(stats_train[3]) / sum(stats_train[1])
        total_train_loss = sum(stats_train[4]) / sum(stats_train[1])

        y_true_all = np.concatenate(all_y_true, axis=0)
        y_prob_all = np.concatenate(all_y_prob, axis=0)
        # 统一标签为 1D
        if y_true_all.ndim == 2:
            y_true_idx = np.argmax(y_true_all, axis=1)
        else:
            y_true_idx = y_true_all.reshape(-1)

        # 统一分数为 1D 正类分数，并得到预测标签
        if y_prob_all.ndim == 2 and y_prob_all.shape[1] == 2:
            y_score = y_prob_all[:, 1].reshape(-1)  # 正类概率列
            y_pred_all = np.argmax(y_prob_all, axis=1)  # 多列时仍可 argmax
        else:
            y_score = y_prob_all.reshape(-1)  # 已是 1D 正类分数
            y_pred_all = (y_score >= 0.5).astype(int)  # 用阈值生成预测

        # —— EDIT —— 全局 ROC-AUC：在拼接后的 y_true_idx / y_score 上计算，避免客户端 NaN 传染
        if np.unique(y_true_idx).size < 2:
            test_auc = float('nan')
        else:
            test_auc = auc_score = auc(*__import__('sklearn.metrics').metrics.roc_curve(y_true_idx, y_score)[
                                        :2])  # 也可直接 roc_auc_score(y_true_idx, y_score)

        acc_score = accuracy_score(y_true_idx, y_pred_all)
        recall = recall_score(y_true_idx, y_pred_all,
                              average="binary" if np.unique(y_true_idx).size == 2 else "macro")
        f1 = f1_score(y_true_idx, y_pred_all,
                      average="binary" if np.unique(y_true_idx).size == 2 else "macro")

        precision, recall_curve, _ = precision_recall_curve(y_true_idx, y_score)
        auprc = auc(recall_curve, precision)

        # total_accs = [a / n for a, n in zip(stats[4], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_online_test_acc.append(online_test_acc)
            self.rs_offline_test_acc.append(offline_test_acc)
            self.rs_total_test_acc.append(total_test_acc)
            self.rs_test_auc.append(test_auc)
            self.rs_test_recall.append(recall)
            self.rs_test_f1.append(f1)
            self.rs_test_auprc.append(auprc)
            self.eval_rounds.append(self.current_round)
        else:
            acc.append(online_test_acc)

        if loss == None:
            self.rs_online_train_loss.append(online_train_loss)
            self.rs_offline_train_loss.append(offline_train_loss)
            self.rs_total_train_loss.append(total_train_loss)
        else:
            loss.append(online_train_loss)

        self.whether_early_stop(f1)

        print("===== Global Evaluation =====")
        # print(f"Averaged Online Train Loss : {online_train_loss:.4f}")
        # print(f"Averaged Offline Train Loss: {offline_train_loss:.4f}")
        # print(f"Averaged Total Train Loss  : {total_train_loss:.4f}")
        # print(f"Averaged Online Test Acc   : {online_test_acc:.4f}")
        # print(f"Averaged Offline Test Acc  : {offline_test_acc:.4f}")
        # print(f"Averaged Total Test Acc    : {total_test_acc:.4f}")
        # print(f"Averaged Test ROC-AUC      : {test_auc:.4f}")
        # print(f"Accuracy (global)          : {acc_score:.4f}")
        # print(f"Recall   (global)          : {recall:.4f}")
        # print(f"F1-score (global)          : {f1:.4f}")
        # print(f"AUPRC    (global)          : {auprc:.4f}")
        # 简化的评估输出
        # round_display = getattr(self, 'current_round', 0)
        print(f"Round {self.current_round}: Acc={acc_score:.3f} Recall={recall:.3f} F1={f1:.4f} AUC={test_auc:.3f}  Losses: Online={online_train_loss:.3f} Offline={offline_train_loss:.3f}")

        # 检测异常情况，决定是否打印详细信息
        show_details = False
        if hasattr(self, 'prev_f1'):
            if abs(f1 - self.prev_f1) > 0.01:  # F1显著变化
                show_details = True
        else:
            show_details = True  # 第一轮显示详细信息

        # 检测其他异常
        if online_train_loss > 2.0 or offline_train_loss > 2.0 or offline_train_loss < 0.2 and offline_test_acc < 0.3:
            show_details = True

        if show_details:
            print(f"  Details: OnlineAcc={online_test_acc:.3f} OfflineAcc={offline_test_acc:.3f} AUPRC={auprc:.3f}")
            print(f"  Losses: Online={online_train_loss:.3f} Offline={offline_train_loss:.3f}")

        self.prev_f1 = f1  # 保存上一轮的F1用于比较


    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:  # 检查还有没有新的客户端需要评估
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        online_correct = []
        offline_correct = []
        total_correct = []
        tot_auc = []  # 本批次总的auc分数
        all_y_true = []   #所有客户端的真实标签
        all_y_prob = []   #所有客户端的预测概率
        for c in self.clients:  # 对于每一个客户端执行下面操作：
            online_test_correct_num, offline_test_correct_num, total_test_correct_num, test_num, auc, y_true, y_prob = c.test_metrics()  # 得到本次预测正确的数量，样本数量，auc值
            online_correct.append(online_test_correct_num * 1.0)  # 将本次预测正确的数量转换成浮点数插入正确数量统计列表
            offline_correct.append(offline_test_correct_num * 1.0)
            total_correct.append(total_test_correct_num * 1.0)

            tot_auc.append(auc * test_num)  # 总auc分数，即之后可以通过每个客户端的样本数量来决定这个客户端对全局的贡献
            num_samples.append(test_num)  # 将本次测试的样本数加入样本总数统计列表

            all_y_true.append(y_true)  # 累加真实标签
            all_y_prob.append(y_prob)  # 累加预测概率
        ids = [c.id for c in self.clients]

        return ids, num_samples, online_correct, offline_correct, total_correct, tot_auc, all_y_true, all_y_prob

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        online_losses_list = []
        offline_losses_list = []
        total_losses_list = []
        for c in self.clients:
            online_losses, offline_losses, total_losses, train_num = c.train_metrics()  # 计算本批次总损失和训练的客户端数量
            num_samples.append(train_num)
            online_losses_list.append(online_losses * 1.0)
            offline_losses_list.append(offline_losses * 1.0)
            total_losses_list.append(total_losses * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, online_losses_list, offline_losses_list, total_losses_list




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

    def init_layer(self):
        self.structure[0][0] = [np.arange(self.num_clients),]

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
        """检测欺诈检测中的梯度冲突（基于FedAWA理论）"""
        if len(self.selected_clients) < 2:
            return False

        # 收集有欺诈样本的客户端
        fraud_clients = []
        normal_only_clients = []

        for client_id in [c.id for c in self.selected_clients]:
            fraud_count = self.get_client_fraud_count(client_id)
            if fraud_count > 0:
                fraud_clients.append(client_id)
            else:
                normal_only_clients.append(client_id)

        # 如果欺诈客户端太少，无法检测有意义的冲突
        if len(fraud_clients) < 2:
            return False

        # 基于客户端更新向量计算冲突
        client_vectors = {}
        for client_id in fraud_clients:
            if hasattr(self, 'client_parameters_history') and client_id < len(self.client_parameters_history):
                if len(self.client_parameters_history[client_id]) > 0:
                    client_vectors[client_id] = self.compute_fraud_aware_client_vector(client_id, layer)

        if len(client_vectors) < 2:
            return False

        # 计算向量间相似度
        similarities = []
        client_ids = list(client_vectors.keys())

        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                vec_i = client_vectors[client_ids[i]]
                vec_j = client_vectors[client_ids[j]]

                # 简化的相似度计算
                if vec_i and vec_j:
                    sim = self.compute_vector_similarity(vec_i, vec_j)
                    similarities.append(sim)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            conflict_detected = avg_similarity < 0.3  # 相似度阈值

            # 记录到层级指标中
            self.layer_metrics[layer]['gradient_conflicts'].append(conflict_detected)

            return conflict_detected

        return False

    def get_client_fraud_count(self, client_id):
        """获取客户端欺诈样本数量（基于你提供的分布）"""
        fraud_counts = {
            0: 0, 1: 2000, 2: 718, 3: 0, 4: 1124, 5: 2452, 6: 2080, 7: 89,
            8: 0, 9: 0, 10: 297, 11: 0, 12: 0, 13: 114, 14: 497, 15: 0,
            16: 0, 17: 5186, 18: 946, 19: 5160
        }
        return fraud_counts.get(client_id, 0)

    def compute_fraud_aware_client_vector(self, client_id, layer):
        """计算欺诈感知的客户端向量"""
        fraud_count = self.get_client_fraud_count(client_id)

        if fraud_count == 0:
            return {'type': 'normal_only', 'weight_factor': 0.1, 'fraud_ratio': 0.0}

        # 基于最近的训练历史
        if hasattr(self, 'client_training_history') and client_id < len(self.client_training_history):
            recent_history = self.client_training_history[client_id][-3:] if self.client_training_history[
                client_id] else []

            if recent_history:
                avg_loss = sum(record.get('loss', 0.5) for record in recent_history) / len(recent_history)
                loss_variance = sum((record.get('loss', 0.5) - avg_loss) ** 2 for record in recent_history) / len(
                    recent_history)

                total_samples = self.get_client_sample_count(client_id)
                fraud_ratio = fraud_count / total_samples if total_samples > 0 else 0.0

                return {
                    'type': 'fraud_aware',
                    'fraud_ratio': fraud_ratio,
                    'avg_loss': avg_loss,
                    'loss_variance': loss_variance,
                    'weight_factor': 1.0 + fraud_ratio  # 欺诈样本多的客户端权重更高
                }

        # 默认返回
        return {'type': 'default', 'fraud_ratio': 0.05, 'weight_factor': 1.0}

    def get_client_sample_count(self, client_id):
        """获取客户端总样本数量"""
        total_counts = {
            0: 97727, 1: 10899, 2: 758, 3: 111428, 4: 1349, 5: 2483, 6: 12373, 7: 2447,
            8: 54438, 9: 6994, 10: 516, 11: 36290, 12: 78195, 13: 11327, 14: 2489, 15: 45714,
            16: 79961, 17: 21658, 18: 1340, 19: 12154
        }
        return total_counts.get(client_id, 1)

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

        # 计算欺诈感知权重
        aggregation_weights = self.compute_fraud_aware_weights()

        # 执行加权聚合
        self.weighted_parameter_aggregation(aggregation_weights)

        print(f"第 {layer} 层使用欺诈感知聚合，权重分布: {[f'{w:.3f}' for w in aggregation_weights[:5]]}")

    def compute_fraud_aware_weights(self):
        """计算基于欺诈样本的聚合权重"""
        weights = []

        for i, client_id in enumerate(self.uploaded_ids):
            fraud_count = self.get_client_fraud_count(client_id)
            sample_count = self.get_client_sample_count(client_id)

            # 基础权重（样本数）
            base_weight = self.uploaded_weights[i]  # 原始的样本比例权重

            if fraud_count == 0:
                # 无欺诈样本的客户端：大幅降低权重
                final_weight = base_weight * 0.2
            else:
                # 有欺诈样本的客户端：根据比例增强权重
                fraud_ratio = fraud_count / sample_count

                # 欺诈率越高，权重增强越多（但有上限）
                enhancement_factor = min(3.0, 1.0 + fraud_ratio * 5.0)
                final_weight = base_weight * enhancement_factor

            weights.append(final_weight)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return weights

    def weighted_parameter_aggregation(self, weights):
        """执行加权参数聚合"""
        if len(self.uploaded_models) == 0:
            return

        # 获取全局模型的状态字典
        global_dict = self.global_model.state_dict()

        # 对每个参数进行加权平均
        for key in global_dict.keys():
            if 'num_batches_tracked' in key:
                continue  # 跳过BatchNorm的跟踪参数

            # 初始化为零张量
            weighted_param = torch.zeros_like(global_dict[key])

            # 加权求和
            for model, weight in zip(self.uploaded_models, weights):
                if key in model.state_dict():
                    weighted_param += weight * model.state_dict()[key].to(self.device)

            # 更新全局模型
            global_dict[key].data.copy_(weighted_param)

    def fraud_detection_convergence_monitor(self, layer, round_idx, metrics):
        """基于欺诈检测理论的收敛监控"""
        current_f1 = metrics.get('f1', 0.0)
        current_auc = metrics.get('auc', 0.5)
        current_recall = metrics.get('recall', 0.0)

        # 初始化层级历史
        if not hasattr(self, 'layer_convergence_history'):
            self.layer_convergence_history = {}
        if layer not in self.layer_convergence_history:
            self.layer_convergence_history[layer] = {'f1': [], 'auc': [], 'recall': []}

        # 记录当前指标
        self.layer_convergence_history[layer]['f1'].append(current_f1)
        self.layer_convergence_history[layer]['auc'].append(current_auc)
        self.layer_convergence_history[layer]['recall'].append(current_recall)

        should_stop = False
        stop_reasons = []

        # 1. 性能连续下降检查
        if len(self.layer_convergence_history[layer]['f1']) >= 3:
            recent_f1 = self.layer_convergence_history[layer]['f1'][-3:]
            if all(recent_f1[i] >= recent_f1[i + 1] for i in range(len(recent_f1) - 1)) and recent_f1[0] - recent_f1[
                -1] > 0.01:
                should_stop = True
                stop_reasons.append("F1连续显著下降")

        # 2. 性能过低检查（针对欺诈检测）
        if round_idx >= 3 and current_f1 < 0.03 and current_auc < 0.55:
            should_stop = True
            stop_reasons.append("性能过低无改善")

        # 3. 梯度冲突过多
        if (hasattr(self, 'layer_metrics') and layer in self.layer_metrics and
                len(self.layer_metrics[layer]['gradient_conflicts']) > 0):
            recent_conflicts = self.layer_metrics[layer]['gradient_conflicts'][-3:]
            if len(recent_conflicts) >= 3 and sum(recent_conflicts) >= 2:
                should_stop = True
                stop_reasons.append("梯度冲突频繁")

        # 4. 欺诈检测特定：召回率过低
        if round_idx >= 5 and current_recall < 0.05:
            should_stop = True
            stop_reasons.append("召回率过低")

        if should_stop:
            print(f"第 {layer} 层轮次 {round_idx} 触发早停: {', '.join(stop_reasons)}")
            print(f"最终指标: F1={current_f1:.4f}, AUC={current_auc:.4f}, Recall={current_recall:.4f}")

        return should_stop

    def adjust_aggregation_strategy(self, layer, mode):
        """根据检测到的问题调整聚合策略"""
        if mode == "conflict_mode":
            # 启用欺诈感知聚合
            self.fraud_aware_aggregation_enabled = True
            # 可以进一步调整其他参数
        else:
            self.fraud_aware_aggregation_enabled = False