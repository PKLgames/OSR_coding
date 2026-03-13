#!/usr/local/miniconda3/envs/py312/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# 启用cudnn benchmark加速
torch.backends.cudnn.benchmark = True

from utils.TAU22 import TAUDataset
import yamnet_PT_inference as yamnet_infer
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from Flow_Models import FlowBasedTissue, FlowBasedCell
from loss_function import TrainDatasetLoss, CalibDatasetLoss

from exam_tool import check_for_anomalies
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# ===============================
# 3. 模型接口定义
# ===============================
class ModelInterface(nn.Module):
    """模型接口基类"""
    
    def __init__(self, 
                 num_known_classes: int = 10):
        """
        初始化模型
        
        参数:
            num_known_classes: 分类类别数
        """
        super().__init__()
        self.num_known_classes = num_known_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'num_known_classes': self.num_known_classes
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

# ===============================
# 4. 具体CNN模型实现
# ===============================

# 门控网络 #
class PolicyNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNet, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)  # 添加Dropout防过拟合
        self.fc2 = nn.Linear(256, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.1)  # 添加Dropout防过拟合
        self.fc3 = nn.Linear(32, out_dim)

    def forward(self, x, temp):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), 0.1))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), 0.1))
        logits = self.fc3(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask


class FinalModel(ModelInterface):
    
    def __init__(self, 
                 num_unknown_classes: int = 8,
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = False):
        super().__init__(num_known_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = "cpu"
        self.feature_model = torch_yamnet(pretrained=False)
        self.num_unknown_classes = num_unknown_classes
        self.num_known_classes = num_known_classes

        # 特征维度
        feature_num = 32
        condition_dim = 0
        self.condition_vector = torch.zeros(condition_dim).to(self.device)

        # 模型参数
        # 加载yamnet预训练模型权重
        if reload_feature_model_pretrained:
            path = 'yamnet.pth'
            state = torch.load(path)
            self.feature_model.load_state_dict(state)

        # # 在训练前微调yamnet
        # for param in self.feature_model.parameters():
        #     param.requires_grad = True

        classifier_cluster_num = 8

        self.fc1 = nn.Sequential(
            nn.Linear(521, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, feature_num),
            nn.BatchNorm1d(feature_num), 
            nn.Tanh()
        )

        # self.classifier = FlowBasedTissue(
        #                     input_dim=feature_num,
        #                     prototype_num=num_known_classes+1, # 加1表示合并未知类
        #                     condition_dim=condition_dim,
        #                     num_coupling_layers=6,
        #                     hidden_dims=[64, 64],
        #                     use_permutation=True,
        #                     permutation_type='fixed',
        #                     cluster_num=classifier_cluster_num # 分类器聚类数
        #                     )
        # self.clusterer = FlowBasedCell(
        #                     input_dim = feature_num,
        #                     condition_dim = condition_dim,
        #                     num_coupling_layers = 5,
        #                     hidden_dims = [64, 64],
        #                     use_permutation = True,
        #                     permutation_type = 'fixed',
        #                     cluster_num = num_unknown_classes
        #                     )
        # 门控网络
        self.num_experts = 5
        self.num_experts_cluster = 5
        self.classifier_experts = nn.ModuleList([FlowBasedTissue(
                                                input_dim=feature_num,
                                                prototype_num=num_known_classes+1, # 加1表示合并未知类
                                                condition_dim=condition_dim,
                                                num_coupling_layers=4,
                                                hidden_dims=[64, 64],
                                                use_permutation=True,
                                                permutation_type='fixed',
                                                cluster_num=classifier_cluster_num # 分类器聚类数
                                                ) for _ in range(self.num_experts)])
        self.clusterer_experts = nn.ModuleList([FlowBasedCell(
                                                input_dim = feature_num,
                                                condition_dim = condition_dim,
                                                num_coupling_layers = 3,
                                                hidden_dims = [64, 64],
                                                use_permutation = True,
                                                permutation_type = 'fixed',
                                                cluster_num = num_unknown_classes
                                                ) for _ in range(self.num_experts_cluster)])
        self.gate1 = PolicyNet(feature_num, self.num_experts)
        self.gate2 = PolicyNet(feature_num, self.num_experts_cluster)

        # 可训练阈值（标量参数）
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use GPU-based feature extraction for better performance
        x = yamnet_infer.waveform_to_log_mel_patches_gpu(x, sample_rate=16000)  # [batch, N, 96, 64] on GPU
        
        feature = self.feature_model(x, to_prob=False)
        feature = self.fc1(feature)
        feature = self.fc2(feature)
        feature = self.fc3(feature) # [batch, feature_num]
        batch_size = feature.size(0)
        # print("feature:", feature.shape, feature.dtype)

        # 检查 feature 张量中的异常值
        # try:
        #     check_for_anomalies(feature, "Problematic Tensor")
        # except ValueError as e:
        #     print(f"Caught error: {e}")

        # preds = self.classifier(feature, self.condition_vector)  # [batch, num_known_classes]
        
        weights1 = self.gate1(feature, 1) # [batch, num_experts]
        weights1 = top_k_gating(weights1, k=self.num_experts) # [batch, num_experts] top-k稀疏化
        
        preds = torch.zeros(batch_size, self.num_known_classes+1, device=self.device) # [batch, num_known_classes+1]
        for i in range(self.num_experts):
            expert_output = self.classifier_experts[i](feature, self.condition_vector)
            preds += weights1[:, i].unsqueeze(1) * expert_output # [batch, num_known_classes+1]

        preds = F.softmax(preds, dim=1)  # [batch, num_known_classes+1]

        # preds = known2unknown_probs(preds)  # [batch, known_classes + 1]
        # 好像会崩溃出现NaN，暂时先不加这个了，后续再调试
        # print("preds:", preds.shape, preds.dtype)

        masked_feature,_ = get_masked_feature(feature, preds)
        # print("masked_feature:", masked_feature.shape, masked_feature.dtype, masked_feature.max(), masked_feature.min())

        #数据增强
        masked_feature_aug = feature_augment(masked_feature)
        
        # masked_preds, log_probs = self.clusterer(masked_feature, self.condition_vector)  # [batch, num_unknown_classes]
        # masked_preds_aug, log_probs_aug = self.clusterer(masked_feature_aug, self.condition_vector)  # [batch, num_unknown_classes]

        weights2 = self.gate2(feature, 1)  # [batch, num_experts_cluster]
        weights2 = top_k_gating(weights2, k=self.num_experts_cluster) # [batch, num_experts_cluster] top-k稀疏化

        masked_preds = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        masked_preds_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        for i in range(self.num_experts_cluster):
            expert_preds, expert_log_probs = self.clusterer_experts[i](masked_feature, self.condition_vector)
            expert_preds_aug, expert_log_probs_aug = self.clusterer_experts[i](masked_feature_aug, self.condition_vector)
            masked_preds += weights2[:, i].unsqueeze(1) * expert_preds
            full_log_probs += weights2[:, i].unsqueeze(1) * expert_log_probs
            masked_preds_aug += weights2[:, i].unsqueeze(1) * expert_preds_aug
            full_log_probs_aug += weights2[:, i].unsqueeze(1) * expert_log_probs_aug

        masked_preds = F.softmax(masked_preds, dim=1) # [batch, num_unknown_classes]
        masked_preds_aug = F.softmax(masked_preds_aug, dim=1) # [batch, num_unknown_classes]

        # print("masked_preds:", masked_preds.shape, masked_preds.dtype)
        # print("masked_preds_aug:", masked_preds_aug.shape, masked_preds_aug.dtype)

        return preds, masked_preds, masked_preds_aug, weights1, weights2


# ===============================
# 5. 训练器类
# ===============================
class Trainer:
    """训练器类"""
    
    def __init__(self,
                model: ModelInterface,
                train_loader: DataLoader,
                calib_loader: DataLoader,
                test_loader: DataLoader,
                train_dataset,
                calib_dataset,
                test_dataset,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            calib_loader: 校准数据加载器
            test_loader: 验证数据加载器
            train_dataset: 训练数据集
            calib_dataset: 校准数据集
            test_dataset: 验证数据集
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.calib_loader = calib_loader
        self.test_loader = test_loader
        
        # # pair_dict: {int(k): v for k, v in self.pair_dict.items()}
        # self.calib_dataset_pair_dict = calib_dataset.pair_dict
        # self.test_dataset_pair_dict = test_dataset.pair_dict
        self.device = device
        
        self.load_balance_weight = 0.0001  # 新增超参数

        # 损失函数和优化器
        # self.criterion = nn.CrossEntropyLoss()
        self.num_known_classes = self.model.num_known_classes  # 已知类数
        num_unknown_classes = self.model.num_unknown_classes  # 未知类数
        self.train_dataset_criterion = TrainDatasetLoss()
        # 改进损失函数参数：增加熵正则权重，启用课程学习
        self.calib_dataset_criterion = CalibDatasetLoss(
            lambda_ps=0.5, 
            lambda_reg=0.2,   # 增加簇平衡正则
            lambda_ent=0.15,   # 增加熵正则防过拟合
            lambda_exprt1=0.1, 
            lambda_exprt2=0.05,
            use_curriculum=True  # 启用课程学习
        )
        # 改进优化器：降低学习率，增加weight_decay防过拟合
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        # 学习率调度器 - 使用更平滑的衰减
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)
        
        # 早停机制参数
        self.patience = 3  # 早停耐心值
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-5)  # 替换StepLR
        
        
        # 训练历史
        self.history = {
            'total_train_loss': [],
            'train_loss': [],
            'calib_loss': [],
            'train_acc': [],
            'calib_acc': [],
            'calib_cluster_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_cluster_acc': []
        }
    
    def train_calib_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, item in enumerate(self.train_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            targets = compress_targets(targets, self.num_known_classes) # 压缩标签为 [batch, known_classes + 1] 形式
            targets = targets.argmax(dim=1).long()  # 转换为类别索引

            # 前向传播
            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)
            # known_outputs: [batch_size, num_known_classes+1] dtype=torch.float32 
            # unknown_outputs: [batch_size, num_unknown_classes] dtype=torch.float32
            # unknown_outputs_aug: [batch_size, num_unknown_classes] dtype=torch.float32
            # full_log_probs: [batch_size, num_unknown_classes] dtype=torch.float32
            # targets: [batch_size,] dtype=torch.long

            train_loss, _ = self.calib_dataset_criterion(known_outputs, unknown_outputs, unknown_outputs_aug, targets, weights1, weights2)

            # 反向传播
            self.optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # # ---- gradient 调试 ----
            # zero_grad_params = [
            #     name for name, p in self.model.named_parameters()
            #     if p.grad is None or torch.all(p.grad == 0)
            # ]
            # if zero_grad_params:
            #     print(
            #         f"Warning: train step {batch_idx} has "
            #         f"{len(zero_grad_params)} zero‑grad params, "
            #         f"first few: {zero_grad_params[:5]}"
            #     )
            # # ------------------------
            
            self.optimizer.step()
            
            # 统计
            train_running_loss += train_loss.item()
            _, predicted = known_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 每100个batch打印一次
            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Train_Loss: {train_loss.item():.4f}')
        
        train_epoch_loss = train_running_loss / len(self.train_loader)
        train_epoch_acc = 100. * train_correct / train_total
        
        calib_running_loss = 0.0
        calib_correct = 0
        calib_total = 0
        calib_cluster_correct = 0
        calib_cluster_total = 1  # 避免除零错误

        # # 初始化，调试用
        # L_r_val_running_loss = 0.0
        # L_known_val_running_loss = 0.0
        # L_unknown_val_running_loss = 0.0
        # L_pairwise_val_running_loss = 0.0
        
        for batch_idx, item in enumerate(self.calib_loader):
            inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
            targets = compress_targets(targets, self.num_known_classes) # 压缩标签为 [batch, known_classes + 1] 形式
            # global_indices = item['idx'].to(self.device)  # 获取全局索引
            targets = targets.argmax(dim=1).long()  # 转换为类别索引

            # 前向传播
            known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)
            # known_outputs: [batch_size, num_known_classes+1] dtype=torch.float32 
            # unknown_outputs: [batch_size, num_unknown_classes] dtype=torch.float32
            # unknown_outputs_aug: [batch_size, num_unknown_classes] dtype=torch.float32
            # full_log_probs: [batch_size, num_unknown_classes] dtype=torch.float32
            # global_indices: [batch_size] dtype=torch.int64
            # targets: [batch_size,] dtype=torch.long

            calib_loss, _ = self.calib_dataset_criterion(known_outputs, unknown_outputs, unknown_outputs_aug, targets, weights1, weights2)

            # 反向传播
            self.optimizer.zero_grad()
            calib_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            
            # 统计
            calib_running_loss += calib_loss.item()
            # 已知类正确率
            calib_total += targets.size(0)
            _, predicted = known_outputs.max(1)
            calib_correct += predicted.eq(targets).sum().item()
            # 未知类正确率
            calib_cluster_total += (unknown_outputs != 0).all(dim=1).sum().item()
            _, predicted_unknown = unknown_outputs.max(1)
            _, predicted_unknown_aug = unknown_outputs_aug.max(1)
            calib_cluster_correct += predicted_unknown.eq(predicted_unknown_aug).sum().item()


            # # 统计各个损失项的值，调试用
            # L_r_val_running_loss += L_r_val.item()
            # L_known_val_running_loss += L_known_val.item()
            # L_unknown_val_running_loss += L_unknown_val.item()
            # L_pairwise_val_running_loss += L_pairwise_val.item()
            
            # 每100个batch打印一次
            if batch_idx % 100 == 99:
                print(f'  Batch: {batch_idx+1}, Calib_Loss: {calib_loss.item():.4f}')
        
        calib_epoch_loss = calib_running_loss / len(self.calib_loader)
        calib_epoch_acc = 100. * calib_correct / calib_total
        calib_epoch_cluster_acc = 100. * calib_cluster_correct / calib_cluster_total

        # # 打印各个损失项的平均值，调试用
        # L_r_val_epoch_loss = L_r_val_running_loss / len(self.calib_loader)
        # L_known_val_epoch_loss = L_known_val_running_loss / len(self.calib_loader)
        # L_unknown_val_epoch_loss = L_unknown_val_running_loss / len(self.calib_loader)
        # L_pairwise_val_epoch_loss = L_pairwise_val_running_loss / len(self.calib_loader)
        # print(f'  Calib Loss Breakdown: L_r: {L_r_val_epoch_loss:.4f}, L_known: {L_known_val_epoch_loss:.4f}, L_unknown: {L_unknown_val_epoch_loss:.4f}, L_pairwise: {L_pairwise_val_epoch_loss:.4f}')

        # 综合训练集和校准集的损失
        epoch_loss = train_epoch_loss + calib_epoch_loss
        print(f'Train Epoch Loss: {epoch_loss:.4f} (Train: {train_epoch_loss:.4f}, Calib: {calib_epoch_loss:.4f})')

        return epoch_loss, train_epoch_loss, calib_epoch_loss, train_epoch_acc, calib_epoch_acc, calib_epoch_cluster_acc

    def test(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        cluster_correct = 0
        cluster_total = 1  # 避免除零错误

        # # 初始化，调试用
        # L_r_val_running_loss = 0.0
        # L_known_val_running_loss = 0.0
        # L_unknown_val_running_loss = 0.0
        # L_pairwise_val_running_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, item in enumerate(self.test_loader):
                inputs, targets = item['source_audio'].to(self.device), item['target'].squeeze(1).to(self.device)
                targets = compress_targets(targets, self.num_known_classes) # 压缩标签为 [batch, known_classes + 1] 形式
                # global_indices = item['idx'].to(self.device)  # 获取全局索引
                targets = targets.argmax(dim=1).long()  # 转换为类别索引

                known_outputs, unknown_outputs, unknown_outputs_aug, weights1, weights2 = self.model(inputs)
                # known_outputs: [batch_size, num_known_classes+1] dtype=torch.float32 
                # unknown_outputs: [batch_size, num_unknown_classes] dtype=torch.float32
                # unknown_outputs_aug: [batch_size, num_unknown_classes] dtype=torch.float32
                # full_log_probs: [batch_size, num_unknown_classes] dtype=torch.float32
                # global_indices: [batch_size] dtype=torch.int64
                # targets: [batch_size,] dtype=torch.long

                loss, _ = self.calib_dataset_criterion(known_outputs, unknown_outputs, unknown_outputs_aug, targets, weights1, weights2)
                
                running_loss += loss.item()
                # 已知类正确率
                total += targets.size(0)
                _, predicted = known_outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                # 未知类正确率
                cluster_total += (unknown_outputs != 0).all(dim=1).sum().item()
                _, predicted_unknown = unknown_outputs.max(1)
                _, predicted_unknown_aug = unknown_outputs_aug.max(1)
                cluster_correct += predicted_unknown.eq(predicted_unknown_aug).sum().item()

                # # 统计各个损失项的值，调试用
                # L_r_val_running_loss += L_r_val.item()
                # L_known_val_running_loss += L_known_val.item()
                # L_unknown_val_running_loss += L_unknown_val.item()
                # L_pairwise_val_running_loss += L_pairwise_val.item()
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        test_cluster_acc = 100. * cluster_correct / cluster_total

        # # 打印各个损失项的平均值，调试用
        # L_r_val_epoch_loss = L_r_val_running_loss / len(self.test_loader)
        # L_known_val_epoch_loss = L_known_val_running_loss / len(self.test_loader)
        # L_unknown_val_epoch_loss = L_unknown_val_running_loss / len(self.test_loader)
        # L_pairwise_val_epoch_loss = L_pairwise_val_running_loss / len(self.test_loader)
        # print(f'  Test Loss Breakdown: L_r: {L_r_val_epoch_loss:.4f}, L_known: {L_known_val_epoch_loss:.4f}, L_unknown: {L_unknown_val_epoch_loss:.4f}, L_pairwise: {L_pairwise_val_epoch_loss:.4f}')

###### test loss不升高，过拟合？ ######
###### 分类和聚类模型/专家间特征融合 ######
###### 改数据增强/校准集加量 ######

        return test_loss, test_acc, test_cluster_acc
    
    def train(self, epochs: int = 10):
        """训练模型"""
        print(f'开始训练，设备: {self.device}')
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # 训练
            total_train_loss, train_loss, calib_loss, train_acc, calib_acc, calib_cluster_acc = self.train_calib_epoch()
            
            # 验证
            test_loss, test_acc, test_cluster_acc = self.test()
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存历史
            self.history['total_train_loss'].append(total_train_loss)
            self.history['train_loss'].append(train_loss)
            self.history['calib_loss'].append(calib_loss)
            self.history['train_acc'].append(train_acc)
            self.history['calib_acc'].append(calib_acc)
            self.history['calib_cluster_acc'].append(calib_cluster_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_cluster_acc'].append(test_cluster_acc)
            
            print(f'总训练 Loss: {total_train_loss:.4f}')
            print(f'训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'校准 Loss: {calib_loss:.4f}, Acc: {calib_acc:.2f}%, Cluster Acc: {calib_cluster_acc:.2f}%')
            print(f'验证 Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, Cluster Acc: {test_cluster_acc:.2f}%')
            
            # 早停检查
            if test_loss < self.best_val_loss:
                self.best_val_loss = test_loss
                self.early_stop_counter = 0
                # 保存最佳模型
                self.model.save('experiment/yamnet_C2MoE/yamnet_C2MoE_best.pth')
                print(f'  >> 保存最佳模型 (Val Loss: {test_loss:.4f})')
            else:
                self.early_stop_counter += 1
                print(f'  >> 早停计数: {self.early_stop_counter}/{self.patience}')
                if self.early_stop_counter >= self.patience:
                    print(f'\n!!! 早停触发，停止训练 !!!')
                    break
    
            # 添加 TensorBoard 观测门控权重分布
            writer.add_histogram('Gate1/Weights', self.model.gate1.fc1.weight, epoch)  # 观测 gate1 第一层权重
            writer.add_histogram('Gate1/Biases', self.model.gate1.fc1.bias, epoch)    # 观测 gate1 第一层偏置
            writer.add_histogram('Gate2/Weights', self.model.gate2.fc1.weight, epoch)  # 观测 gate2 第一层权重
            writer.add_histogram('Gate2/Biases', self.model.gate2.fc1.bias, epoch)    # 观测 gate2 第一层偏置

            # 可选：观测梯度（如果需要）
            if self.model.gate1.fc1.weight.grad is not None:
                writer.add_histogram('Gate1/Grad_Weights', self.model.gate1.fc1.weight.grad, epoch)
            if self.model.gate2.fc1.weight.grad is not None:
                writer.add_histogram('Gate2/Grad_Weights', self.model.gate2.fc1.weight.grad, epoch)

    def plot_history(self,from_epochs: int, to_epochs: int):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失曲线
        axes[0].plot(self.history['total_train_loss'], label='Total train Loss')
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['calib_loss'], label='Calib Loss')
        axes[0].plot(self.history['test_loss'], label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and testidation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制准确率曲线
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['calib_acc'], label='Calib Acc')
        axes[1].plot(self.history['calib_cluster_acc'], label='Calib Cluster Acc')
        axes[1].plot(self.history['test_acc'], label='Test Acc')
        axes[1].plot(self.history['test_cluster_acc'], label='Test Cluster Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and testidation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'experiment/yamnet_C2MoE/yamnet_C2MoE_trained_from{str(from_epochs)}to{str(to_epochs)}.png')

# ===============================
# 6. 辅助函数：生成筛选特征
# ===============================
def get_masked_feature(feature: torch.Tensor,
                        preds: torch.Tensor) -> torch.Tensor:
    """
    生成筛选后的特征：已知类特征归零，未知类特征保留
    
    参数:
        feature: 原始特征，形状为 [batch_size, feature_num]
        preds: 类别概率，形状为 [batch_size, num_sum_classes+1]，表示每个样本的类别概率
    
    返回:
        masked_feature: [batch_size, feature_num] 的筛选特征
    """
    # 获取每个样本的最大概率值
    max_vals, max_indices = torch.max(preds, dim=1)  # [batch_size]

    # 判断最大值是否出现在最后一列（即未知类）
    # 注意：torch.max 返回的是第一个最大值的索引，但如果最后一列等于最大值，也可能不是它
    # 更安全的做法：检查最后一列是否等于最大值
    is_unknown = (preds[:, -1] >= max_vals)  # [batch_size], 允许并列最大

    # 转为 float 掩码：未知类 = 1, 已知类 = 0
    mask = is_unknown.float()  # [batch_size]

    # 3. 扩展掩码到特征和预测维度
    mask_f = mask.unsqueeze(1).expand_as(feature)      # [batch_size, feature_num]
    mask_p = mask.unsqueeze(1).expand_as(preds)        # [batch_size, num_sum_classes]

    # 4. 应用掩码：已知类置零，未知类保留
    masked_feature = feature * mask_f
    masked_preds = preds * mask_p

    return masked_feature, masked_preds


def known2unknown_probs(probs, epsilon=1e-8):
    batch_size, k_num = probs.shape
    
    # 1. 获取最大值和次大值及其索引
    sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
    p_max = sorted_probs[:, 0:1]       # [B, 1]
    p_second = sorted_probs[:, 1:2]    # [B, 1]
    
    # 处理只有一个类别的极端情况 (k_num=1)
    if k_num == 1:
        p_second = torch.zeros_like(p_max)
    
    # 2. 计算相对突出度 (Relative Prominence)
    # 防止除以0
    denominator = p_max + epsilon
    margin_ratio = (p_max - p_second) / denominator  # 范围 [0, 1]
    
    # 3. 计算未知类概率 (Adaptive Unknown Probability)
    sensitivity = 2.0 # 调节灵敏度，越大对“相近”越敏感
    uncertainty_factor = torch.sigmoid((1.0 - margin_ratio) * sensitivity - (sensitivity/2)) 
    # 上面的sigmoid中心点在 margin_ratio = 0.5 处，即 p_second = 0.5 * p_max
    
    p_unknown = p_max * uncertainty_factor
    
    # 4. 调整已知类概率
    # 原始概率 * 突出度因子，确保当不确定时，已知类概率降低
    known_probs_scaled = probs * (1.0 - uncertainty_factor)
    
    # 5. 归一化拼接输出 [batch, known_num + 1]
    output_probs = F.softmax(torch.cat([known_probs_scaled, p_unknown], dim=1),dim=1)
    
    return output_probs


def feature_augment(
    features: torch.Tensor,
    p_apply: float = 1.0,        # 每个样本以概率 p_apply 应用增强（否则原样）
    aug_list: list = None) -> torch.Tensor:
    """
    对 [B, D] 音频特征 batch 中每个样本独立地随机应用一种数据增强。
    
    Args:
        features: Tensor of shape [batch_size, feature_dim], e.g., [64, 32]
        p_apply: 每个样本被增强的概率（用于控制增强强度）
        aug_list: 可选，指定增强子集，如 ['noise', 'time_mask', 'shift']
    
    Returns:
        augmented_features: same shape as input
    """
    if aug_list is None:
        # aug_list = [
        #     'none',
        #     'noise',
        #     'time_mask',
        #     'scale',
        #     'shift',
        #     'channel_dropout',
        #     'impulse',
        #     'gain_clip'
        # ]
        aug_list = [
            'none',
            'noise',
            'time_mask',
            'scale',
            'invert'
        ]

    batch_size, feat_dim = features.shape
    device = features.device
    augmented = features.clone()

    for i in range(batch_size):
        if random.random() > p_apply:
            continue  # 跳过增强

        sample = features[i]  # [feat_dim]
        aug_type = random.choice(aug_list)

        if aug_type == 'none':
            augmented[i] = sample
        elif aug_type == 'noise':
            noise = torch.randn_like(sample, device=device) * 0.01
            augmented[i] = sample + noise
        elif aug_type == 'time_mask':
            mask_len = random.randint(1, min(6, feat_dim // 3))
            start = random.randint(0, feat_dim - mask_len)
            aug_sample = sample.clone()
            aug_sample[start:start + mask_len] = 0.0
            augmented[i] = aug_sample
        elif aug_type == 'scale':
            scale = torch.empty(1, device=device).uniform_(0.85, 1.15).item()
            augmented[i] = sample * scale
        elif aug_type == 'invert':
            augmented[i] = -sample
        elif aug_type == 'shift':
            shift = random.randint(-feat_dim // 4, feat_dim // 4)
            augmented[i] = torch.roll(sample, shifts=shift, dims=0)
        elif aug_type == 'channel_dropout':
            # 随机丢弃 1~3 个维度
            n_drop = random.randint(1, min(3, feat_dim // 4))
            drop_idx = torch.randperm(feat_dim, device=device)[:n_drop]
            aug_sample = sample.clone()
            aug_sample[drop_idx] = 0.0
            augmented[i] = aug_sample
        elif aug_type == 'band_mask':
            # 保留中间一段，两边置零
            center = random.randint(feat_dim // 4, 3 * feat_dim // 4)
            half_width = random.randint(4, feat_dim // 2)
            left = max(0, center - half_width)
            right = min(feat_dim, center + half_width)
            aug_sample = torch.zeros_like(sample, device=device)
            aug_sample[left:right] = sample[left:right]
            augmented[i] = aug_sample
        elif aug_type == 'impulse':
            # 在 1~3 个随机位置加尖峰
            n_imp = random.randint(1, 3)
            pos = torch.randperm(feat_dim, device=device)[:n_imp]
            impulse = torch.randn(n_imp, device=device) * 0.1
            aug_sample = sample.clone()
            aug_sample[pos] += impulse
            augmented[i] = aug_sample
        elif aug_type == 'gain_clip':
            gain = torch.empty(1, device=device).uniform_(0.8, 1.3).item()
            clipped = torch.clamp(sample * gain, min=-3.0, max=3.0)
            augmented[i] = clipped
        else:
            raise ValueError(f"Unknown augmentation: {aug_type}")

    return augmented

def compress_targets(
    targets: torch.Tensor,
    num_known_classes: int
    ) -> torch.Tensor:
    """
    将原始 one-hot 标签压缩为 [known_classes + 1] 形式，其中最后一类代表所有未知类的合并。
    
    参数:
        targets: [batch_size, num_sum_classes], one-hot
        num_known_classes: int, 表示已知类的数量 K（即索引 0 ~ K-1 是已知类）
    
    返回:
        new_targets: [batch_size, num_known_classes + 1], one-hot
    """
    K = num_known_classes

    # 1. 提取已知类部分
    known_part = targets[:, :K]  # [B, K]

    # 2. 提取未知类部分（从 K 到 end）
    unknown_part = targets[:, K:]  # [B, total_classes - K]

    # 3. 判断是否属于未知类：只要 unknown_part 中有 1，就为 1
    # 因为是 one-hot，可以用 sum 或 any
    is_unknown = unknown_part.sum(dim=1, keepdim=True)  # [B, 1], 值为 0 或 1

    # 4. 拼接：known_part + is_unknown
    new_targets = torch.cat([known_part, is_unknown], dim=1)  # [B, K + 1]

    return new_targets

def top_k_gating(weights: torch.Tensor, k: int = 3, temperature: float = 1.0) -> torch.Tensor:
    """
    对门控权重执行Top-K选择，只保留每行中前k个最大的值，其余置零
    
    Args:
        weights: 门控权重 [batch_size, num_experts]
        k: 选择的专家数量
        temperature: 温度参数，控制选择的集中程度
    
    Returns:
        处理后的权重 [batch_size, num_experts]，每行只有k个非零值
    """
    # 应用温度缩放
    scaled_weights = weights / temperature
    
    # 创建掩码：找出每行中top-k的位置
    top_k_values, top_k_indices = torch.topk(scaled_weights, k=k, dim=-1, sorted=False)
    
    # 创建全零张量
    masked_weights = torch.zeros_like(weights)
    
    # 使用scatter_将top-k的值放回原位置
    batch_indices = torch.arange(weights.size(0), device=weights.device).unsqueeze(-1).expand(-1, k)
    masked_weights.scatter_(1, top_k_indices, torch.ones_like(top_k_values))
    
    # 将原始权重与掩码相乘，保留原始值
    result = weights * masked_weights
    
    # 重新归一化，确保每行和为1
    result = F.normalize(result, p=1, dim=-1)
    
    return result

# ===============================
# 7. 主程序 - 使用示例
# ===============================
def main():
    # 设置随机种子,以确保结果可复现
    torch.manual_seed(42)
    origin_train_epochs = 0
    train_epochs = 20

    num_known_classes = 6
    num_sum_classes = 14
    num_unknown_classes = num_sum_classes - num_known_classes

    
    # 1. 创建数据集
    print("加载数据集...")
    train_dataset = TAUDataset(split='train')# 训练集只包含已知类
    calib_dataset = TAUDataset(split='calib')# 校准集包含已知类和未知类
    test_dataset = TAUDataset(split='test')# 验证集包含已知类和未知类
    
    # 2. 创建数据加载器 (优化GPU利用率)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,           # 增加数据加载并行数
        pin_memory=True,         # 加速CPU→GPU传输
        prefetch_factor=4,      # 预加载批次
        persistent_workers=True  # 保持worker进程
    )
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    # 显示数据集信息
    # print(f"已知类别: {train_dataset.get_class_names()}")
    # print(f"全部类别: {test_dataset.get_class_names()}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"校准集样本数: {len(calib_dataset)}")
    print(f"验证集样本数: {len(test_dataset)}")
    
    # 3. 创建并加载模型
    # 更改模型结构需重加载模型：1.设置reload_feature_model_pretrained=True 2.注释掉模型加载代码 3.设置train_epochs=0 
    print("\n创建模型...")
    model = FinalModel(num_unknown_classes=num_unknown_classes, num_known_classes=num_known_classes, 
                        reload_feature_model_pretrained=True)
    print("\n加载模型...")
    # 由于PolicyNet结构变化导致参数不兼容，从头训练
    # model = FinalModel.load(path='yamnet_C2MoE_origin_0.pth')
    # model = FinalModel.load(path='experiment/yamnet_C2MoE/yamnet_C2MoE_trained_10.pth')
    print("从头训练新模型（PolicyNet结构已更新含Dropout）")

    
    # 显示模型结构
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 创建训练器并训练
    print("\n从第",origin_train_epochs,"轮创建训练器...")
    print("\n开始训练...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        calib_loader=calib_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset
    )
    
    # 创建保存目录
    os.makedirs('experiment/yamnet_C2MoE', exist_ok=True)

    # 训练模型
    trainer.train(epochs=train_epochs)
    
    # 5. 绘制训练历史
    trainer.plot_history(from_epochs=origin_train_epochs,
                            to_epochs=origin_train_epochs+train_epochs)
    
    # 6. 保存模型
    model.save('experiment/yamnet_C2MoE/'+
                'yamnet_C2MoE_trained_'+str(origin_train_epochs+train_epochs)+'.pth')
    print(f'\n模型已保存到: experiment/yamnet_C2MoE/'+
                'yamnet_C2MoE_trained_'+str(origin_train_epochs+train_epochs)+'.pth')
    
    # 7. 测试模型加载
    print("\n测试模型加载...")
    loaded_model = FinalModel.load(path = 'experiment/yamnet_C2MoE/'+
                'yamnet_C2MoE_trained_'+str(origin_train_epochs+train_epochs)+'.pth')
    print("模型加载成功!")
    
    return trainer, loaded_model

# ===============================
# 8. 运行
# ===============================
if __name__ == "__main__":
    # 运行完整示例
    trainer, model = main()
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    