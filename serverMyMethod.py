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
    def train(self):   #启用
        """多层级训练主循环"""
        # 初始化第0层
        self.structure[0][0] = [np.arange(self.num_clients), copy.deepcopy(self.global_model).state_dict()]

        for layer in range(self.max_layers):
            print(f"\n{'=' * 50}")
            print(f"开始第 {layer} 层训练")
            print(f"活跃客户端: {sorted(list(self.active_clients_set))}")
            print(f"{'=' * 50}")

            # 训练当前层
            self.train_current_layer(layer)

            # # 如果不是最后一层，生成下一层聚类
            # if layer < self.max_layers - 1 and len(self.active_clients_set) > self.min_clients_per_cluster:
            #     self.generate_next_layer_clusters(layer)
            # else:
            #     print(f"在第 {layer} 层结束训练")
            #     break

        # 原有的保存和绘图代码保持不变
        self.save_results()
        self.save_global_model()
        self.plot_metrics()

    def train_current_layer(self, layer):   #启用
        """训练指定层级的联邦学习"""

        print(f"\n开始第 {layer} 层训练")
        print(f"活跃客户端: {[client.id for client in self.selected_clients]}")

        # 渐进式层级训练设置
        self.progressive_layer_training(layer)

        layer_rounds = self.layer_rounds[layer] if layer < len(self.layer_rounds) else 15

        self.selected_clients = self.select_active_clients()

        if layer == 0:
            clusters = [[client.id for client in self.selected_clients]]
            print(f"第 {layer} 层采用全局聚合")
        else:
            # 高层级使用梯度相似度聚类
            client_updates = self.collect_client_updates()
            clusters = self.adaptive_layer_clustering(client_updates, layer)
            print(f"第 {layer} 层生成了 {len(clusters)} 个聚类")
            for i, cluster in enumerate(clusters):
                print(f"聚类 {i}: 客户端 {cluster}")

        # client_updates = self.collect_client_updates()
        # clusters = self.adaptive_layer_clustering(client_updates, layer)
        # print(f"第 {layer} 层生成了 {len(clusters)} 个聚类")
        # for i, cluster in enumerate(clusters):
        #     print(f"聚类 {i}: 客户端 {cluster}")

        self.initialize_cluster_models(layer, clusters)
        self.send_models_with_classifier(layer, clusters)

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
            noisy_client_ratios = self.get_noisy_pos_ratios(self.selected_clients)# 在每轮循环内部，为本轮被选中的客户端计算自适应 alpha
            adaptive_alphas = self.compute_adaptive_alpha(noisy_client_ratios)

            #梯度冲突检测
            if round_idx > 0:
                conflict_detected = self.detect_fraud_gradient_conflicts(layer)
                if conflict_detected:
                    self.adjust_aggregation_strategy(layer, "conflict_mode")
                    print(f"第 {layer} 层检测到梯度冲突，调整聚合策略")

            # 客户端训练
            self.uploaded_models = {cluster_id: [] for cluster_id in range(len(clusters))}
            self.uploaded_weights = {cluster_id: [] for cluster_id in range(len(clusters))}
            self.uploaded_ids = {cluster_id: [] for cluster_id in range(len(clusters))}

            for cluster_id, cluster_clients_ids in enumerate(clusters):
                # 获取当前聚类的客户端对象列表
                cluster_clients = [self.clients[cid] for cid in cluster_clients_ids]

                # 训练聚类内的客户端
                for client in cluster_clients:
                    # 调用客户端的训练函数
                    # if client.id == 0 or client.id == 1:
                        client.train_layered(layer, round_idx, self.num_clients, adaptive_alphas[client.id])
                        # print("len（client.train_samples）的值为：",len(client.train_samples))
                        # print("len（client.train_samples）的类型为：",type(len(client.train_samples)))
                        # 在客户端训练后，立即将模型接收到服务器
                        self.uploaded_models[cluster_id].append(client.model)
                        self.uploaded_weights[cluster_id].append(client.train_samples)
                        self.uploaded_ids[cluster_id].append(client.id)

                # client.train(round_idx,self.num_clients)
            # self.receive_models_with_classifier(clusters)
            print(f"第 {layer} 层轮次 {round_idx} 完成客户端训练")
            # 检查上传的模型是否有异常
            if hasattr(self, 'fraud_aware_aggregation_enabled') and self.fraud_aware_aggregation_enabled:
                self.fraud_aware_layer_aggregation(layer,clusters)
            else:
                # self.aggregate_parameters_by_layer(layer)
                print("train_current_layer:no aggregate function!")
            if round_idx % self.eval_gap == 0:
                self.current_round = round_idx
                self.evaluate_clusters_in_layer(layer, clusters)
            #理论驱动的收敛监控
            metrics = {
                'f1': self.rs_test_f1[-1] if self.rs_test_f1 else 0.0,
                'auc': self.rs_test_auc[-1] if self.rs_test_auc else 0.5,
                'recall': self.rs_test_recall[-1] if self.rs_test_recall else 0.0
            }
#------------------------------------------------------------------------------------#
            # if self.fraud_detection_convergence_monitor(layer, round_idx, metrics):
            #     print(f"第 {layer} 层在轮次 {round_idx} 早停")
            #     break
# ------------------------------------------------------------------------------------#

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

    def select_active_clients(self):   #启用
        """选择活跃客户端进行训练"""
        active_list = list(self.active_clients_set)
        selected_num = min(self.current_num_join_clients, len(active_list))
        selected_indices = random.sample(active_list, selected_num)
        return [self.clients[i] for i in selected_indices]

    def initialize_cluster_models(self, layer, clusters):   #启用
        """
        为指定层级的每个聚类初始化模型。
        """
        # 确保当前层级的字典存在
        if layer not in self.structure:
            self.structure[layer] = {}

        for cluster_id, client_ids in enumerate(clusters):
            # 仅当该聚类模型不存在时才初始化
            if cluster_id not in self.structure[layer]:
                # 获取聚类中第一个客户端的初始模型参数作为起点
                first_client_id = client_ids[0]
                initial_model_params = copy.deepcopy(self.clients[first_client_id].model).state_dict()

                # 将初始化好的模型参数存入对应的聚类键下
                self.structure[layer][cluster_id] = [client_ids, initial_model_params]
                print(f"服务器: 第 {layer} 层，聚类 {cluster_id} 模型参数初始化完成。")






    def extract_update_signature(self, client_id):   #启用
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



    def freeze_current_layer_parameters(self, layer):   #启用
        """冻结当前层的参数"""
        if layer not in self.structure:
            print("freeze_current_layer_parameters:layer not in self.structure")
            return

        # 遍历当前层级的每个聚类模型
        for cluster_id, (_, cluster_model) in self.structure[layer].items():
            # 1. 创建一个临时的全局模型对象
            # 这一步是必要的，因为 `requires_grad` 属性是 nn.Parameter 的，不是 state_dict 的。
            temp_model = copy.deepcopy(self.global_model)

            # 2. 将存储的参数加载到临时模型上
            temp_model.load_state_dict(cluster_model)

            # 3. 遍历临时模型的所有参数并全部冻结
            for param in temp_model.parameters():
                param.requires_grad = False

            # 4. 将修改后的参数再保存回 self.structure
            self.structure[layer][cluster_id][1] = temp_model.state_dict()

        print(f"已冻结第 {layer} 层的参数")

    def update_client_performance_tracking(self):   #启用
        """更新客户端性能跟踪"""
        # 这里使用最新的F1分数作为性能指标
        current_f1 = self.rs_test_f1[-1] if self.rs_test_f1 else 0.0

        for client_id in self.active_clients_set:
            if client_id not in self.client_performance_history:
                self.client_performance_history[client_id] = []
            self.client_performance_history[client_id].append(current_f1)




    def send_models_with_classifier(self, layer, clusters):   #启用
        assert (len(self.clients) > 0)
        # 构建一个客户端ID到聚类ID的映射
        client_to_cluster_map = {}
        for cluster_idx, cluster_clients_ids in enumerate(clusters):
            for client_id in cluster_clients_ids:
                client_to_cluster_map[client_id] = cluster_idx

        for client in self.clients:
            start_time = time.time()
            try:
                cluster_idx = client_to_cluster_map[client.id]
            except KeyError:
                print(f"警告: 客户端 {client.id} 不在当前聚类列表中。")
                continue
            model_to_send = self.structure[layer][cluster_idx][1]
            filtered_model_to_send = {
                key: value
                for key, value in model_to_send.items()
                if "bn" not in key  # 筛选掉所有包含"bn"的参数，包括weight, bias, running_mean, running_var
            }
            client.set_parameters_with_classifier(
                filtered_model_to_send,  # 只发送结构信息
                self.classifier_list,
                layer  # 传递当前层级
            )
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)




    def evaluate_clusters_in_layer(self, layer, clusters):   #启用
        """
        对指定层级的每个聚类模型进行评估。
        """
        print(f"\n--- 第 {layer} 层开始分聚类评估 ---")
        print(f"\n--- 全局训练轮数第 {self.current_round} 轮---")

        # 遍历当前层级的每个聚类
        for cluster_id, _ in enumerate(clusters):
            # 确保该聚类模型存在
            if layer in self.structure and cluster_id in self.structure[layer]:
                # 获取当前聚类的最新聚合模型参数
                model_params = self.structure[layer][cluster_id][1]

                # 将聚类模型参数加载到 self.global_model 中进行评估
                with torch.no_grad():
                    for key, param in model_params.items():
                        if key in self.global_model.state_dict():
                            self.global_model.state_dict()[key].data.copy_(param)

                print(f"服务器: 正在评估第 {layer} 层、聚类 {cluster_id} 的模型...")

                # 执行评估，并打印结果
                self.evaluate()

        print(f"--- 第 {layer} 层分聚类评估完成 ---")


    def evaluate(self, acc=None, loss=None):   #启用
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
            batch_size=self.batch_size,
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
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("-------------------------------------------\n")

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'auprc': auprc,
            'recall': recall
        }




    def log_round_summary(self, round_num):   #启用
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

    def plot_metrics(self):   #启用
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

    def detect_fraud_gradient_conflicts(self, layer):   #启用
        """
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



    def fraud_aware_layer_aggregation(self, layer, clusters):   #启用
        """
        基于客户端模型的AUPRC进行加权聚合。
        """
        if not hasattr(self, 'uploaded_models') or not self.uploaded_models:
            print("警告: 没有上传的模型数据，无法进行聚合。")
            return

            # 遍历每个聚类，进行聚合
        for cluster_id, uploaded_cluster_models in self.uploaded_models.items():
            if not uploaded_cluster_models:
                print(f"警告: 聚类 {cluster_id} 没有上传模型，跳过聚合。")
                continue

            # 步骤1: 评估每个客户端模型，并计算其AUPRC加权
            auprcs = {}
            client_updates = []
            total_weight_sum = 0

            # 获取当前聚类的全局模型参数，结构为 self.structure[layer][cluster_idx][1]
            current_cluster_params = self.structure[layer][cluster_id][1]
            uploaded_cluster_ids = self.uploaded_ids[cluster_id]

            # 用于平滑的全局先验（例如全局验证集的AUPRC）
            global_ap_prior = 0.1
            k = 50  # 平滑强度
            lam = 0.5  # 样本数融合比例



            for client_id, client_model in zip(uploaded_cluster_ids, uploaded_cluster_models):
                model_params = client_model.state_dict()
                print(f"正在评估客户端 {client_id} 模型...")
                metrics = self.evaluate_client_model_before_aggregate(model_params, cluster_id, layer, client_id)
                if metrics is None:
                    print(f"警告: 客户端 {client_id} 的评估结果为 None，跳过该客户端。")
                    continue  # 跳过该客户端

                print(f"评估结果: {metrics}")
                raw_auprc = metrics.get('AUPRC', 0.0)
                n_samples = metrics.get('NumSamples', 0)

                # 平滑后的 AUPRC
                ap_smooth = (raw_auprc * n_samples + global_ap_prior * k) / (n_samples + k)

                # 融合样本数的权重
                weight = lam * n_samples + (1 - lam) * ap_smooth

                auprcs[client_id] = weight
                total_weight_sum += weight
                client_updates.append((weight, model_params))
            if total_weight_sum == 0:
                print(f"警告: 聚类 {cluster_id} 的权重总和为0，无法进行聚合。")
                continue

            aggregated_params = {
                key: torch.zeros_like(value).to(self.device)
                for key, value in current_cluster_params.items()
            }

            with torch.no_grad():
                for weight, model_params in client_updates:
                    normalized_weight = weight / total_weight_sum
                    for key in aggregated_params:
                        if key in model_params and 'bn' not in key:
                            aggregated_params[key] += normalized_weight * model_params[key].to(self.device)
                        elif 'bn' in key:
                            aggregated_params[key] += current_cluster_params[key].to(self.device)

                for key, param in aggregated_params.items():
                    if key in self.structure[layer][cluster_id][1]:
                        self.structure[layer][cluster_id][1][key].data.copy_(param.cpu())

            print(f"服务器: 第 {layer} 层，聚类 {cluster_id} 模型聚合完成，并已更新。")
            client_id_list = uploaded_cluster_ids
            print(f"   聚合权重 (平滑+样本融合): {[f'{auprcs.get(cid, 0):.3f}' for cid in client_id_list[:5]]}")

        # # 为每个上传模型的客户端计算欺诈感知权重
            # aggregation_weights = []
            # for client in cluster_clients:
            #     weight = self.compute_fraud_aware_weights(client)
            #     aggregation_weights.append(weight)
            #
            #
            # aggregation_weights = []
            # for client in cluster_clients:
            #     # 直接使用客户端上传的 F1-Score 作为权重
            #     # 注意：你需要确保客户端对象在服务器端仍然保持其状态
            #     # 或者在上传参数时，客户端同时上传其 F1 值
            #     if hasattr(client, 'local_f1_score'):
            #         weight = client.local_f1_score
            #     else:
            #         # 如果无法获取，使用默认权重，例如 1.0
            #         weight = 1.0
            #
            #     # 确保权重是正数，避免除以0
            #     aggregation_weights.append(max(0.0, weight))
            #
            # total_weight = sum(aggregation_weights)
            # if total_weight > 0:
            #     aggregation_weights = [w / total_weight for w in aggregation_weights]
            # else:
            #     # 如果权重总和为0，则使用平均权重
            #     aggregation_weights = [1.0 / len(uploaded_cluster_models)] * len(uploaded_cluster_models)
            #
            # # 执行加权聚合，并得到聚合后的参数字典
            # aggregated_params = self.weighted_parameter_aggregation(uploaded_cluster_models, aggregation_weights)
            #
            # # 将聚合参数应用到 self.structure 中对应的聚类模型
            # if aggregated_params:
            #     with torch.no_grad():
            #         # 检查 self.structure 结构
            #         if layer not in self.structure:
            #             self.structure[layer] = {}
            #         if cluster_id not in self.structure[layer] or self.structure[layer][cluster_id][1] is None:
            #             # 如果聚类模型不存在，则初始化
            #             self.structure[layer][cluster_id] = [uploaded_client_ids, aggregated_params]
            #         else:
            #             # 更新已存在的模型参数
            #             current_cluster_params = self.structure[layer][cluster_id][1]
            #             for key, param in aggregated_params.items():
            #                 if key in current_cluster_params:
            #                     current_cluster_params[key].data.copy_(param)
            #
            #     print(f"服务器: 第 {layer} 层，聚类 {cluster_id} 模型聚合完成，并已更新。")
            #     print(f"   权重分布: {[f'{w:.3f}' for w in aggregation_weights[:5]]}")

#--------------------------------------------------↑↑↑-------------------------------------------------------

    # def compute_fraud_aware_weights(self, client):
    #     """
    #     为聚合或为某些策略分配权重。不得使用精确样本数。
    #     使用等级映射到权重的策略（可自定义映射或学习得到）。
    #     返回单个浮点数权重。
    #     """
    #     level = self.get_client_fraud_level(client)
    #     # 简单策略（可调整或用函数映射）
    #     # 低欺诈 -> 普通权重
    #     # 中等 -> 轻微增加（因为中等样本可能更具信息性）
    #     # 高 -> 适当增加（因为高欺诈样本对模型影响力重要）
    #     if level == 0:
    #         return 1.0
    #     elif level == 1:
    #         return 1.2
    #     elif level == 2:
    #         return 1.5
    #     else:
    #         return 1.0


    def weighted_parameter_aggregation(self, models, weights, trim_ratio=0.2):   #启用
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

    def fraud_detection_convergence_monitor(self, layer, round_idx, metrics, patience=3, min_delta=1e-4):   #启用
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

    def adjust_aggregation_strategy(self, layer, mode):   #启用
        """根据检测到的问题调整聚合策略"""
        if mode == "conflict_mode":
            # 启用欺诈感知聚合
            self.fraud_aware_aggregation_enabled = True
            # 可以进一步调整其他参数
        else:
            self.fraud_aware_aggregation_enabled = False

    def get_client_fraud_level(self, client): #启用
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
    def compute_gradient_similarity(self, client_updates):   #启用
        """计算客户端之间的相似度"""
        similarities = {}
        client_ids = list(client_updates.keys())

        # === 调试打印 ===
        if not client_ids:
            print("警告: 聚类输入为空，无法计算相似度。")
            return {}
        print(f"聚类输入客户端数量: {len(client_ids)}")
        sample_client_id = client_ids[0]
        sample_update = client_updates[sample_client_id]
        # print(f"示例客户端 {sample_client_id} 的更新类型: {type(sample_update)}")
        # if isinstance(sample_update, dict):
        #     print(f"示例更新字典的键: {list(sample_update.keys())}")


        for i, client_a in enumerate(client_ids):
            similarities[client_a] = {}
            grad_a = self.flatten_param(client_updates[client_a])

            # === 调试打印 ===
            print(f"客户端 {client_a} 的展平梯度范数: {torch.norm(grad_a).item():.4f}")

            for j, client_b in enumerate(client_ids):
                if i <= j:
                    grad_b = self.flatten_param(client_updates[client_b])
                    # 使用余弦相似度
                    similarity = torch.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0))
                    similarities[client_a][client_b] = similarity.item()
                    if client_b not in similarities:
                        similarities[client_b] = {}
                    similarities[client_b][client_a] = similarity.item()

        return similarities

    def flatten_param(self, model_params):   #启用
        """将模型参数字典展平为一维向量，并计算其范数作为梯度范数"""
        flattened = []
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                # 这里的param是state_dict中的张量，没有requires_grad属性
                flattened.append(param.flatten().to(self.device))

        if not flattened:
            return torch.zeros(1, device=self.device)

        return torch.cat(flattened)

    def adaptive_layer_clustering(self, client_updates, layer_idx):   #启用
        """自适应层级聚类"""
        similarities = self.compute_gradient_similarity(client_updates)

        # 根据层级调整相似度阈值
        similarity_threshold = 0.7 - 0.1 * layer_idx  # 随层级递减

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

    def progressive_layer_training(self, layer_idx):   #启用
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

    def collect_client_updates(self):   #启用
        """收集客户端模型更新用于聚类分析"""
        updates = {}
        for client in self.clients:
            updates[client.id] = client.latest_model_params
        return updates



    def compute_adaptive_alpha(self, noisy_client_ratios):   #启用
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

    def get_noisy_pos_ratios(self, clients): # 接收客户端对象列表    #启用
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

    def evaluate_client_model_before_aggregate(self, model_params, cluster_id, layer,client_id):
        """
        使用服务器的本地验证集对客户端模型进行评估。
        """
        print("开始加载模型参数...")
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(model_params)
        print("模型参数加载成功")
        temp_model.to(self.device)
        temp_model.eval()

        test_loader = DataLoader(
            self.global_test_data,
            batch_size=self.batch_size,
            shuffle=False
        )

        y_true = []
        y_pred_probs = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = temp_model(inputs)
                probs = torch.sigmoid(outputs)

                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

        y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
        metrics = {}
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['AUPRC'] = average_precision_score(y_true, y_pred_probs)
        num_samples = None
        try:
            # 先定位 client_id 在该 cluster 中的 index
            idx = self.uploaded_ids[cluster_id].index(client_id)
        except Exception:
            idx = None
            print("evaluate_client_model_before_aggregate:try except wrong!")

        if idx is not None:
            try:
                num_samples = self.uploaded_weights[cluster_id][idx]
            except Exception:
                num_samples = None
                print("evaluate_client_model_before_aggregate:try except wrong!")
            # 如果 uploaded_weights 不可用，再尝试从 uploaded_models 中读取（如果那一项是 dict 且包含 'num_samples'）
        if num_samples is None and idx is not None:
            try:
                uploaded_entry = self.uploaded_models[cluster_id][idx]
                if isinstance(uploaded_entry, dict):
                    num_samples = uploaded_entry.get('num_samples', None)
            except Exception:
                num_samples = None
                print("evaluate_client_model_before_aggregate:try except wrong!")
            # 最后回退到全局验证集的样本数（兼容旧逻辑）
        if num_samples is None:
            num_samples = len(y_true)

        try:
            metrics['NumSamples'] = int(num_samples)
        except Exception:
            metrics['NumSamples'] = len(y_true)
            print("evaluate_client_model_before_aggregate:try except wrong!")
        return metrics
