import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TrainDatasetLoss(nn.Module):
    def __init__(self):
        super(TrainDatasetLoss, self).__init__()
        self.lambda_ps = 0.5
        self.lambda_reg = 0.1

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        gamma: torch.Tensor,
        gamma_aug: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        device = classifier_log_probs.device
        batch_size = classifier_log_probs.size(0)
        num_unknown_classes = classifier_log_probs.size(1) - 1
        
        # ========== L_osc: 开放集分类损失 ==========
        L_osc = F.cross_entropy(classifier_log_probs, targets, reduction='mean')
        
        # clusterer相关损失
        L_ps = torch.tensor(0.0, device=device)
        L_reg = torch.tensor(0.0, device=device)
        if gamma is not None and gamma.numel() > 0:
            if gamma_aug is not None:
                L_ps = F.mse_loss(gamma, gamma_aug, reduction='mean')
            avg = gamma.mean(dim=0)
            L_reg = F.kl_div((avg + 1e-8).log(),
                             torch.full_like(avg, 1.0 / avg.size(0)),
                             reduction='sum')
        
        # 训练时没有未知类，所以我们暂时不使用 clusterer 相关损失
        total_loss = L_osc

        loss_dict = {
            'L_osc': L_osc.detach(),
            'L_ps': L_ps.detach(),
            'L_reg': L_reg.detach(),
        }
        
        return total_loss, loss_dict

class CalibDatasetLoss(nn.Module):
    """
    改进后的损失函数模块。
    改进策略：
    1. 引入熵正则化 (Entropy Regularization) 防止过拟合和模糊聚类。
    2. 严格基于 Mask 过滤全零行，仅有效样本参与聚类损失计算。
    3. 仅依赖 gamma (模型输出) 和 gamma_aug (增强视图)，不依赖 targets 进行聚类约束。
    4. 添加课程学习策略：根据训练进度动态调整损失权重
    """

    def __init__(
        self,
        lambda_ps: float = 0.5,      # 一致性损失权重 (Consistency Loss)
        lambda_reg: float = 0.2,     # 簇平衡正则权重 (Balance Regularization)
        lambda_ent: float = 0.1,     # 熵正则权重 (Entropy Regularization) - 提高以防止过拟合
        lambda_exprt1: float = 0.1,  # 专家权重平衡损失权重 (Expert Balance Loss)
        lambda_exprt2: float = 0.05,  # 第二个专家权重平衡损失权重 (Second Expert Balance Loss)
        eps: float = 1e-8,
        use_curriculum: bool = True  # 是否使用课程学习
    ):
        super(CalibDatasetLoss, self).__init__()
        self.lambda_ps = lambda_ps
        self.lambda_reg = lambda_reg
        self.lambda_ent = lambda_ent
        self.lambda_exprt1 = lambda_exprt1
        self.lambda_exprt2 = lambda_exprt2
        self.eps = eps
        self.use_curriculum = use_curriculum
        self.current_epoch = 0  # 课程学习进度

    def set_epoch(self, epoch: int):
        """设置当前epoch，用于课程学习"""
        self.current_epoch = epoch

    def get_curriculum_weights(self, total_epochs: int = 30) -> dict:
        """
        课程学习权重：根据训练进度动态调整
        策略：
        - 早期(0-30%): 主要关注分类损失 L_osc
        - 中期(30-70%): 逐渐加入聚类损失
        - 后期(70-100%): 全部损失参与
        """
        if not self.use_curriculum:
            return {
                'lambda_ps': self.lambda_ps,
                'lambda_reg': self.lambda_reg,
                'lambda_ent': self.lambda_ent,
                'lambda_exprt1': self.lambda_exprt1,
                'lambda_exprt2': self.lambda_exprt2
            }
        
        progress = self.current_epoch / total_epochs
        
        if progress < 0.3:
            # 早期：主要关注分类
            scale = progress / 0.3  # 0 -> 1
            return {
                'lambda_ps': self.lambda_ps * scale * 0.1,
                'lambda_reg': self.lambda_reg * scale * 0.1,
                'lambda_ent': self.lambda_ent * scale * 0.5,  # 熵正则保持一定权重防止过拟合
                'lambda_exprt1': self.lambda_exprt1 * scale,
                'lambda_exprt2': self.lambda_exprt2 * scale
            }
        elif progress < 0.7:
            # 中期：逐渐加入聚类损失
            scale = (progress - 0.3) / 0.4  # 0 -> 1
            return {
                'lambda_ps': self.lambda_ps * (0.1 + scale * 0.5),
                'lambda_reg': self.lambda_reg * (0.1 + scale * 0.5),
                'lambda_ent': self.lambda_ent * (0.5 + scale * 0.3),
                'lambda_exprt1': self.lambda_exprt1,
                'lambda_exprt2': self.lambda_exprt2
            }
        else:
            # 后期：全部损失
            return {
                'lambda_ps': self.lambda_ps,
                'lambda_reg': self.lambda_reg,
                'lambda_ent': self.lambda_ent,
                'lambda_exprt1': self.lambda_exprt1,
                'lambda_exprt2': self.lambda_exprt2
            }

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        gamma: torch.Tensor,
        gamma_aug: torch.Tensor,
        targets: torch.Tensor,
        weights1: torch.Tensor, 
        weights2: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        epoch: int = 0,
        total_epochs: int = 30
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            classifier_log_probs: (batch_size, num_known + 1)
            gamma: (batch_size, cluster_num) - 原始视图的后验概率
            gamma_aug: (batch_size, cluster_num) - 增强视图的后验概率
            targets: (batch_size,) - 仅用于 L_osc
            weights1, weights2: (batch_size,) - 专家权重
            log_probs: (batch_size, cluster_num) - 原始视图的对数似然 (含 -inf 标记无效行)
            epoch: 当前训练轮次
            total_epochs: 总训练轮次
        """
        device = classifier_log_probs.device
        batch_size = classifier_log_probs.size(0)
        
        # 更新epoch用于课程学习
        self.set_epoch(epoch)
        
        # 获取当前课程学习权重
        curr_weights = self.get_curriculum_weights(total_epochs)
        
        # ========== 1. L_osc: 开放集分类损失 (监督部分) ==========
        # 添加标签平滑以防止过拟合
        L_osc = F.cross_entropy(classifier_log_probs, targets, reduction='mean', label_smoothing=0.1)
        
        # ========== 准备聚类损失的 Mask ==========
        if log_probs is not None:
            is_invalid = torch.isneginf(log_probs).all(dim=1)
            valid_mask = ~is_invalid
        else:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            
        n_valid = valid_mask.sum()

        L_ps = torch.tensor(0.0, device=device)
        L_reg = torch.tensor(0.0, device=device)
        L_ent = torch.tensor(0.0, device=device)

        if n_valid > 0:
            gamma_valid = gamma[valid_mask]
            gamma_aug_valid = gamma_aug[valid_mask]

            # ========== 2. L_ps: 视图一致性损失 ==========
            L_ps = F.mse_loss(gamma_valid, gamma_aug_valid, reduction='mean')

            # ========== 3. L_reg: 簇平衡正则化 ==========
            avg_gamma = gamma_valid.mean(dim=0)
            uniform_dist = torch.full_like(avg_gamma, 1.0 / avg_gamma.size(0))
            L_reg = F.kl_div((avg_gamma + self.eps).log(), uniform_dist, reduction='sum')

            # ========== 4. L_ent: 熵正则化 - 防过拟合关键 ==========
            log_gamma_valid = torch.log(gamma_valid + self.eps)
            entropy_per_sample = -torch.sum(gamma_valid * log_gamma_valid, dim=1)
            L_ent = torch.mean(entropy_per_sample)

        # ========== 5. L_exprt: 专家权重平衡损失 ==========
        L_exprt1 = self.compute_expert_balance_loss(weights1)
        L_exprt2 = self.compute_expert_balance_loss(weights2)
        
        # ========== 总损失 - 使用课程学习权重 ==========
        total_loss = L_osc + \
                     curr_weights['lambda_ps'] * L_ps + \
                     curr_weights['lambda_reg'] * L_reg + \
                     curr_weights['lambda_ent'] * L_ent + \
                     curr_weights['lambda_exprt1'] * L_exprt1 + \
                     curr_weights['lambda_exprt2'] * L_exprt2

        loss_dict = {
            'L_osc': L_osc.detach(),
            'L_ps': L_ps.detach(),
            'L_reg': L_reg.detach(),
            'L_ent': L_ent.detach(),
            'L_exprt1': L_exprt1.detach(),
            'L_exprt2': L_exprt2.detach(),
            'n_valid_samples': float(n_valid),
            'curr_lambda_ps': curr_weights['lambda_ps'],
            'curr_lambda_ent': curr_weights['lambda_ent']
        }
        
        return total_loss, loss_dict

    def compute_expert_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算专家使用均衡损失
        """
        # 添加小epsilon防止log(0)
        eps = 1e-8
        avg_usage = torch.mean(weights, dim=0)
        
        # 使用负熵作为均衡性度量
        entropy_loss = torch.sum(avg_usage * torch.log(avg_usage + eps))
        
        # L2距离到均匀分布
        uniform_dist = torch.ones_like(avg_usage) / avg_usage.size(0)
        l2_balance_loss = torch.norm(avg_usage - uniform_dist, p=2)
        
        # 组合
        balance_loss = -entropy_loss + 0.1 * l2_balance_loss
        
        return balance_loss
