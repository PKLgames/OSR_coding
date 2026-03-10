import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TrainDatasetLoss(nn.Module):
    def __init__(
        self,
    ):
        super(TrainDatasetLoss, self).__init__()
        # clusterer-related loss weights; default values can be adjusted
        self.lambda_ps = 0.5
        self.lambda_reg = 0.1

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        gamma: torch.Tensor,
        gamma_aug: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失
        参数:
        ----------
        classifier_log_probs : torch.Tensor
            分类器的对数概率输出 (batch_size, num_unknown_classes + 1)
        targets : torch.Tensor
            真实标签 (batch_size,) 已知类为 [0, ..., num_unknown_classes-1]，未知类为 num_unknown_classes
        gamma : torch.Tensor
            未知样本的簇后验责任值 (n_unknown, cluster_num)，仅在存在未知样本时使用
        gamma_aug : torch.Tensor
            对应增强视图的责任值 (n_unknown, cluster_num)，仅在存在未知样本且 lambda_ps > 0 时使用

        返回:
        ----------
        total_loss : torch.Tensor
            总损失标量
        loss_dict : dict
            各分量损失 {'L_osc'}
        """
        device = classifier_log_probs.device
        batch_size = classifier_log_probs.size(0)
        num_unknown_classes = classifier_log_probs.size(1) - 1  # 最后一维是未知类
        
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
        total_loss = L_osc# + self.lambda_ps * L_ps + self.lambda_reg * L_reg

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
    """

    def __init__(
        self,
        lambda_ps: float = 0.5,      # 一致性损失权重 (Consistency Loss)
        lambda_reg: float = 0.2,     # 簇平衡正则权重 (Balance Regularization)
        lambda_ent: float = 0.1,    # 熵正则权重 (Entropy Regularization) - 新增，防过拟合关键
        eps: float = 1e-8
    ):
        super(CalibDatasetLoss, self).__init__()
        self.lambda_ps = lambda_ps
        self.lambda_reg = lambda_reg
        self.lambda_ent = lambda_ent
        self.eps = eps

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        gamma: torch.Tensor,
        gamma_aug: torch.Tensor,
        targets: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None # 接收来自 FlowBasedClusterer 的 full_log_probs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            classifier_log_probs: (batch_size, num_known + 1)
            gamma: (batch_size, cluster_num) - 原始视图的后验概率
            gamma_aug: (batch_size, cluster_num) - 增强视图的后验概率
            targets: (batch_size,) - 仅用于 L_osc
            log_probs: (batch_size, cluster_num) - 原始视图的对数似然 (含 -inf 标记无效行)
        """
        device = classifier_log_probs.device
        batch_size = classifier_log_probs.size(0)
        num_unknown_classes = classifier_log_probs.size(1) - 1
        
        # ========== 1. L_osc: 开放集分类损失 (监督部分) ==========
        L_osc = F.cross_entropy(classifier_log_probs, targets, reduction='mean')
        
        # ========== 准备聚类损失的 Mask ==========
        # 核心逻辑：如果 log_probs 中存在 -inf，说明该行是全零无效样本
        # 我们利用这一点构建 valid_mask，确保只有真正数据参与聚类损失
        if log_probs is not None:
            # 检查每一行是否包含 -inf (或者所有列都是 -inf)
            # 在 FlowBasedClusterer 中，无效行的所有 log_prob 都被设为 -inf
            is_invalid = torch.isneginf(log_probs).all(dim=1) # (batch_size,)
            valid_mask = ~is_invalid
        else:
            # 如果没有传入 log_probs，退化为检查 gamma 是否全为均匀分布 (较弱)
            # 或者假设所有样本有效 (不推荐)
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            
        n_valid = valid_mask.sum()

        L_ps = torch.tensor(0.0, device=device)
        L_reg = torch.tensor(0.0, device=device)
        L_ent = torch.tensor(0.0, device=device)

        if n_valid > 0:
            # 提取有效样本的数据
            gamma_valid = gamma[valid_mask]       # (n_valid, K)
            gamma_aug_valid = gamma_aug[valid_mask] # (n_valid, K)

            # ========== 2. L_ps: 视图一致性损失 (Contrastive/Consistency) ==========
            # 强制原始视图和增强视图的聚类分配一致
            L_ps = F.mse_loss(gamma_valid, gamma_aug_valid, reduction='mean')

            # ========== 3. L_reg: 簇平衡正则化 (Cluster Balance) ==========
            # 防止某个簇占据所有样本，强制簇分布趋向均匀
            avg_gamma = gamma_valid.mean(dim=0) # (K,)
            uniform_dist = torch.full_like(avg_gamma, 1.0 / avg_gamma.size(0))
            # KL(P || Uniform)
            L_reg = F.kl_div((avg_gamma + self.eps).log(), uniform_dist, reduction='sum')

            # ========== 4. L_ent: 熵正则化 (Entropy Regularization) - 防过拟合关键 ==========
            # 最小化预测分布的熵，鼓励"硬"聚类 (Hard Clustering)
            # 如果模型输出模糊 (如 [0.25, 0.25, 0.25, 0.25])，熵很大，Loss 变大
            # 如果模型输出确定 (如 [1, 0, 0, 0])，熵为 0，Loss 变小
            # 这能有效防止模型在训练初期坍塌到均匀分布，或在噪声上过拟合出模糊边界
            # 计算每个样本的分布熵: - sum(p * log(p))
            # 加 eps 防止 log(0)
            log_gamma_valid = torch.log(gamma_valid + self.eps)
            entropy_per_sample = -torch.sum(gamma_valid * log_gamma_valid, dim=1) # (n_valid,)
            L_ent = torch.mean(entropy_per_sample)

        # ========== 总损失 ==========
        total_loss = L_osc + \
                     self.lambda_ps * L_ps + \
                     self.lambda_reg * L_reg + \
                     self.lambda_ent * L_ent

        loss_dict = {
            'L_osc': L_osc.detach(),
            'L_ps': L_ps.detach(),
            'L_reg': L_reg.detach(),
            'L_ent': L_ent.detach(),
            'n_valid_samples': float(n_valid)
        }
        
        return total_loss, loss_dict









# class CalibDatasetLoss(nn.Module):
#     """
#     基于标准化流的未知类别发现损失函数模块。
#     支持：
#     - 开放集分类损失 L_osc（使用 FlowModelClassier）
#     - 未知样本对比聚类损失 L_ps（使用 FlowBasedClusterer）
#     - 混合权重/责任值均匀正则化 L_reg

#     输入说明：
#     - classifier_log_probs: (batch_size, num_unknown_classes + 1)
#         最后一列为“未知类”原型的对数似然
#     - targets: (batch_size,) 
#         真实标签：已知类为 [0, ..., num_unknown_classes-1]，未知类为 num_unknown_classes
#     - gamma: (n_unknown, cluster_num)
#         未知样本的簇后验责任值（来自 FlowBasedClusterer）
#     - gamma_aug: (n_unknown, cluster_num)
#         对应增强视图的责任值（用于 L_ps）
#     """

#     def __init__(
#         self,
#         lambda_ps: float = 0.5,
#         lambda_reg: float = 0.1,
#     ):
#         """
#         初始化损失模块
        
#         参数:
#         ----------
#         lambda_ps : float
#             对比损失 L_ps 的权重
#         lambda_reg : float
#             正则化损失 L_reg 的权重
#         """
#         super(CalibDatasetLoss, self).__init__()
#         self.lambda_ps = lambda_ps
#         self.lambda_reg = lambda_reg

#     def forward(
#         self,
#         classifier_log_probs: torch.Tensor,
#         gamma: torch.Tensor,
#         gamma_aug: torch.Tensor,
#         targets: torch.Tensor,
#     ) -> Tuple[torch.Tensor, dict]:
#         """
#         计算总损失
#         参数:
#         ----------
#         classifier_log_probs : torch.Tensor
#             分类器的对数概率输出 (batch_size, num_unknown_classes + 1)
#         targets : torch.Tensor
#             真实标签 (batch_size,) 已知类为 [0, ..., num_unknown_classes-1]，未知类为 num_unknown_classes
#         gamma : torch.Tensor
#             未知样本的簇后验责任值 (n_unknown, cluster_num)，仅在存在未知样本时使用
#         gamma_aug : torch.Tensor
#             对应增强视图的责任值 (n_unknown, cluster_num)，仅在存在未知样本且 lambda_ps > 0 时使用

#         返回:
#         ----------
#         total_loss : torch.Tensor
#             总损失标量
#         loss_dict : dict
#             各分量损失 {'L_osc', 'L_ps', 'L_reg'}
#         """
#         device = classifier_log_probs.device
#         batch_size = classifier_log_probs.size(0)
#         num_unknown_classes = classifier_log_probs.size(1) - 1  # 最后一维是未知类
        
#         # ========== L_osc: 开放集分类损失 ==========
#         L_osc = F.cross_entropy(classifier_log_probs, targets, reduction='mean')
        
#         # clusterer相关损失
#         L_ps = torch.tensor(0.0, device=device)
#         L_reg = torch.tensor(0.0, device=device)
#         if gamma is not None and gamma.numel() > 0:
#             if gamma_aug is not None:
#                 L_ps = F.mse_loss(gamma, gamma_aug, reduction='mean')
#             avg = gamma.mean(dim=0)
#             L_reg = F.kl_div((avg + 1e-8).log(),
#                              torch.full_like(avg, 1.0 / avg.size(0)),
#                              reduction='sum')

#         total_loss = L_osc + self.lambda_ps * L_ps + self.lambda_reg * L_reg

#         loss_dict = {
#             'L_osc': L_osc.detach(),
#             'L_ps': L_ps.detach(),
#             'L_reg': L_reg.detach(),
#         }
        
#         return total_loss, loss_dict


#     def L_reg_cal(self, gamma: torch.Tensor) -> torch.Tensor:
#         """
#         计算均匀性正则化损失
        
#         参数:
#         ----------
#         gamma : torch.Tensor
#             未知样本的簇后验责任值 (n_unknown, cluster_num)
        
#         返回:
#         ----------
#         L_reg : torch.Tensor
#             均匀性正则化损失标量
#         """
#         avg_gamma = gamma.mean(dim=0)  # (cluster_num,)
#         uniform = torch.full_like(avg_gamma, 1.0 / avg_gamma.size(0))
#         # 为了数值稳定性，避免对零取 log
#         avg_gamma_safe = avg_gamma + 1e-8
#         L_reg = F.kl_div(
#             avg_gamma_safe.log(),
#             uniform,
#             reduction='sum'
#         )
        
#         return L_reg