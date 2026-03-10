import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import Reversable_Function.cINN as cINN

class PrototypeFlowModel(nn.Module):
    """
    条件可逆神经网络（cINN）的组合模型，使用cluster_num个cINN模型按权重相加
    """
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int=0, 
                 num_coupling_layers: int = 12,
                 hidden_dims: List[int] = [256, 256],
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 cluster_num: int = 6):
        """
        初始化组合模型
        
        参数:
        ----------
        input_dim : int
            输入向量的维度
        condition_dim : int
            条件向量的维度
        num_coupling_layers : int
            耦合层的数量
        hidden_dims : List[int]
            子网络的隐藏层维度
        use_permutation : bool
            是否在耦合层之间使用置换操作
        permutation_type : str
            置换类型: 'fixed'（固定置换）或 'learnable'（可学习置换）
        cluster_num : int
            簇数量
        """
        super(PrototypeFlowModel, self).__init__()
        
        # 创建cluster_num个模型
        self.cluster_num = cluster_num
        self.cluster_model = nn.ModuleList([
            cINN.ConditionalINN(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type
            ) for _ in range(cluster_num)
        ])
        
        # 创建权重参数，用于对cINN模型的输出进行加权
        self.weights = nn.Parameter(torch.ones(cluster_num) / cluster_num)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：从数据空间到隐变量空间
        
        参数:
        ----------
        x : torch.Tensor
            输入数据，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (condition_dim, )
        compute_jacobian : bool
            是否计算雅可比行列式的对数
        
        返回:
        ----------
        z : torch.Tensor
            隐变量，形状为 (batch_size, input_dim)
        log_det_jacobian : Optional[torch.Tensor]
            总雅可比行列式的对数
        """
        # 计算每个cINN模型的输出
        z_list = []
        log_det_list = []
        
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        for i in range(self.cluster_num):
            z, log_det = self.cluster_model[i](x, c_expand, compute_jacobian=compute_jacobian)
            z_list.append(z)
            log_det_list.append(log_det)
        
        # 将所有输出按权重相加
        z = torch.stack(z_list, dim=0)  # (cluster_num, batch_size, input_dim)
        log_det = torch.stack(log_det_list, dim=0) if log_det_list[0] is not None else None
        
        # 对z和log_det进行加权求和
        z = torch.sum(z * self.weights.view(-1, 1, 1), dim=0)
        
        if log_det is not None:
            log_det = torch.sum(log_det * self.weights.view(-1, 1), dim=0)
        
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        逆变换：从隐变量空间到数据空间
        
        参数:
        ----------
        z : torch.Tensor
            隐变量，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (condition_dim, )
        
        返回:
        ----------
        x : torch.Tensor
            重建的数据
        """
        # 计算每个模型的逆变换
        x_list = []
        c_expand = c.repeat(z.shape[0], 1).to(z.device)
        
        for i in range(self.cluster_num):
            x = self.cluster_model[i].inverse(z, c_expand)
            x_list.append(x)
        
        # 将所有输出按权重相加
        x = torch.stack(x_list, dim=0)  # (cluster_num, batch_size, input_dim)
        x = torch.sum(x * self.weights.view(-1, 1, 1), dim=0)
        
        return x

    def sample(self, c: torch.Tensor, num_samples: int = 1, 
               prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        从条件分布中采样
        
        参数:
        ----------
        c : torch.Tensor
            条件向量，形状为 (condition_dim,)
        num_samples : int
            每个条件向量的采样数量
        prior_dist : Optional[torch.distributions.Distribution]
            先验分布，默认为标准正态分布
        
        返回:
        ----------
        samples : torch.Tensor
            采样数据，形状为 (batch_size * num_samples, input_dim) 或 (num_samples, input_dim)
        """
        # 处理输入条件向量的形状
        if c.dim() == 1:
            c_expand = c.unsqueeze(0)
                
        # 从每个cINN模型中采样
        samples_list = []
        
        for i in range(self.cluster_num):
            samples = self.cluster_model[i].sample(c_expand, num_samples, prior_dist)
            samples_list.append(samples)
        
        # 将所有采样结果按权重相加
        samples = torch.stack(samples_list, dim=0)  # (cluster_num, batch_size, num_samples, input_dim)
        samples = torch.sum(samples * self.weights.view(-1, 1, 1, 1), dim=0)
        
        return samples

    def log_prob(self, x: torch.Tensor, c: torch.Tensor, 
                 prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        计算数据的对数似然
        
        参数:
        ----------
        x : torch.Tensor
            数据，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (condition_dim,)
        prior_dist : Optional[torch.distributions.Distribution]
            先验分布，默认为标准正态分布
        
        返回:
        ----------
        log_prob : torch.Tensor
            数据的对数似然，形状为 (batch_size,)
        """
        # 计算每个cINN模型的对数似然
        log_prob_list = []
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        
        for i in range(self.cluster_num):
            log_prob = self.cluster_model[i].log_prob(x, c, prior_dist)
            log_prob_list.append(log_prob)
        
        # 将所有对数似然按权重相加
        log_prob = torch.stack(log_prob_list, dim=0)  # (cluster_num, batch_size)
        log_prob = torch.sum(log_prob * self.weights.view(-1, 1), dim=0)
        
        return log_prob

class FlowBasedTissue(nn.Module):
    """
    由prototype_num个PrototypeFlowModel组成的原型模型，输出每个原型模型的对数似然
    """
    def __init__(self, 
                 input_dim: int, 
                 prototype_num: int,
                 condition_dim: int=0, 
                 num_coupling_layers: int = 12,
                 hidden_dims: List[int] = [256, 256],
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 cluster_num: int = 10):
        """
        初始化原型模型
        
        参数:
        ----------
        input_dim : int
            输入向量的维度
        condition_dim : int
            条件向量的维度
        prototype_num : int
            原型模型的数量
        num_coupling_layers : int
            耦合层的数量（用于每个PrototypeFlowModel）
        hidden_dims : List[int]
            子网络的隐藏层维度（用于每个PrototypeFlowModel）
        use_permutation : bool
            是否在耦合层之间使用置换操作（用于每个PrototypeFlowModel）
        permutation_type : str
            置换类型: 'fixed'（固定置换）或 'learnable'（可学习置换）（用于每个PrototypeFlowModel）
        cluster_num : int
            簇数量（用于每个PrototypeFlowModel）
        """
        super(FlowBasedTissue, self).__init__()
        
        self.prototype_num = prototype_num
        self.prototype_models = nn.ModuleList([
            PrototypeFlowModel(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type,
                cluster_num=cluster_num
            ) for _ in range(prototype_num)
        ])
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算每个原型模型的对数似然
        
        参数:
        ----------
        x : torch.Tensor
            输入数据，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (condition_dim,)
        
        返回:
        ----------
        log_probs : torch.Tensor
            形状为 (batch_size, prototype_num) 的张量，
            每个元素表示输入样本属于对应原型模型的对数似然
        """
        log_probs_list = []
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        
        # 为每个原型模型计算对数似然
        for i in range(self.prototype_num):
            # 使用默认标准正态分布（prior_dist=None）
            log_prob = self.prototype_models[i].log_prob(x, c)
            log_probs_list.append(log_prob)
        
        # 将所有对数似然堆叠成 (prototype_num, batch_size)
        log_probs = torch.stack(log_probs_list, dim=0)  # (prototype_num, batch_size)
        
        # 转置为 (batch_size, prototype_num)
        return log_probs.t()


class FlowBasedCell(nn.Module):
    """
    基于标准化流的聚类器，用于未知类发现。
    输入: (batch_size, input_dim)
    输出: (batch_size, cluster_num) —— 每个样本属于各未知簇的后验概率 gamma_ic
    自动跳过全零行（视为无效样本）
    """
    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        num_coupling_layers: int = 6,
        hidden_dims: List[int] = [128, 128],
        use_permutation: bool = True,
        permutation_type: str = 'fixed',
        cluster_num: int = 4,  # 未知簇数量
        # 正则化超参数
    ):
        super(FlowBasedCell, self).__init__()
        self.input_dim = input_dim
        self.cluster_num = cluster_num
        # 正则化超参数
        self.entropy_weight = 0.1  # 控制聚类硬度的熵正则权重
        self.kl_weight = 1.0       # KL散度正则权重
        self.balance_weight = 0.5  # 簇平衡正则权重
        self.init_noise_std = 0.01 # 初始化时的噪声标准差

        # cluster_num 个流模型，每个代表一个未知簇 q'_c
        self.flows = nn.ModuleList([
            cINN.ConditionalINN(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type
            ) for _ in range(cluster_num)
        ])

        # 2. 初始化混合权重 logits
        # 初始化为均匀分布的对数 (log(1/K))，并添加微小噪声打破对称性
        initial_logits = torch.log(torch.ones(cluster_num) / cluster_num)
        noise = torch.randn_like(initial_logits) * self.init_noise_std
        self.mix_logits = nn.Parameter(initial_logits + noise)

        # 应用权重初始化策略
        self._init_weights()

    def _init_weights(self):
        """
        初始化模型权重。
        对于 Flow 模型，通常内部已有 ActNorm 或类似机制，但这里确保 mix_logits 状态良好。
        如果 cINN 支持 reset_parameters，可以在这里调用。
        """
        # 示例：如果 cINN 有内部线性层需要特定初始化，可在此遍历
        for flow in self.flows:
            if hasattr(flow, 'reset_parameters'):
                flow.reset_parameters()
        
        # 重新确认 mix_logits 的噪声注入（防止加载预训练时覆盖）
        if self.training:
            with torch.no_grad():
                noise = torch.randn_like(self.mix_logits) * self.init_noise_std
                self.mix_logits.add_(noise)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
            c: (condition_dim,) # 条件向量，当前未使用, 应根据batch_size扩展为c_valid
            prior_dist: 可选的先验分布，默认为标准正态分布
        Returns:
            (gamma, full_log_probs)
            gamma: (batch_size, cluster_num)，每行表示对应样本属于各未知簇的后验概率
            full_log_probs: (batch_size, cluster_num)，无效样本行用 -inf 填充
        """
        device = x.device
        batch_size = x.shape[0]

        # Step 1: 找出非全零行（有效样本）
        is_valid = ~torch.all(x == 0, dim=1)  # (batch_size,)
        valid_indices = torch.where(is_valid)[0]
        x_valid = x[valid_indices]  # (n_valid, input_dim)
        n_valid = x_valid.shape[0]

        # 处理条件向量 c
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.shape[0] == 1:
            c_valid = c.expand(n_valid, -1)
        else:
            c_valid = c[valid_indices]

        # 初始化输出 gamma
        gamma = torch.zeros(batch_size, self.cluster_num, device=device)
        full_log_probs = torch.full((batch_size, self.cluster_num), float("-inf"), device=device)

        if n_valid == 0:
            # 全是无效样本，返回均匀分布
            gamma[:] = 1.0 / self.cluster_num
            return gamma.detach(), full_log_probs

        # Step 2: 计算每个流模型在有效样本上的 log_prob
        log_probs_list = []
        for i in range(self.cluster_num):
            log_p = self.flows[i].log_prob(x_valid, c_valid)  # (n_valid,)
            log_probs_list.append(log_p)

        log_probs = torch.stack(log_probs_list, dim=1)  # (n_valid, cluster_num)

        # Step 3: 获取混合权重 pi'_c = softmax(mix_logits)
        log_pi = F.log_softmax(self.mix_logits, dim=0)  # (cluster_num,)
        log_pi = log_pi.unsqueeze(0)  # (1, cluster_num)

        # Step 4: 计算加权 log-likelihood: log(pi_c * q_c(x)) = log_pi_c + log_q_c(x)
        weighted_log_prob = log_probs + log_pi  # (n_valid, cluster_num)

        # Step 5: 计算责任值 gamma_ic = softmax(weighted_log_prob)
        gamma_valid = F.softmax(weighted_log_prob, dim=1)  # (n_valid, cluster_num)

        # Step 6: 填回原 batch 位置
        gamma[valid_indices] = gamma_valid
        full_log_probs[valid_indices] = log_probs

        # 对无效样本（全零行），设为均匀分布（可选：也可设为 0，但 uniform 更合理）
        invalid_mask = ~is_valid
        if invalid_mask.any():
            gamma[invalid_mask] = 1.0 / self.cluster_num
        
        return gamma, full_log_probs

    # helper: 计算 L_total = - sum_i sum_c gamma_ic * log_q_ic（对有效样本求和）
    # def compute_L_total(self, gamma: torch.Tensor, log_q: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     if valid_mask is None:
    #         valid_mask = ~(torch.isneginf(log_q).all(dim=1))
    #     if valid_mask.sum() == 0:
    #         return torch.tensor(0.0, device=gamma.device)
    #     g = gamma[valid_mask]
    #     lq = log_q[valid_mask]
    #     loss = -(g * lq).sum()
    #     return loss


    def get_mix_weights(self) -> torch.Tensor:
        """返回当前混合权重 pi'_c"""
        return F.softmax(self.mix_logits, dim=0)
        
    def predict_clusters(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """返回每个样本最可能的簇索引"""
        gamma, _ = self.forward(x, c)
        return torch.argmax(gamma, dim=1)






# 示例
if __name__ == "__main__":
    # 设置超参数
    input_dim = 32
    condition_dim = 0
    batch_size = 32
    prototype_num = 10
    
    # 创建模型
    classifier_model = FlowBasedTissue(
        input_dim=input_dim,
        prototype_num=prototype_num,
        condition_dim=condition_dim,
        num_coupling_layers=3,
        hidden_dims=[64, 64],
        use_permutation=True,
        permutation_type='fixed',
        cluster_num=10
    )

    clusterer_model = FlowBasedCell(
        input_dim = input_dim,
        batch_size = batch_size,
        condition_dim = 0,
        num_coupling_layers = 4,
        hidden_dims = [64, 64],
        use_permutation = True,
        permutation_type = 'fixed',
        cluster_num = 4
    )
    
    print(f"classifier_model模型结构:")
    # print(classifier_model)
    print(f"总参数量: {sum(p.numel() for p in classifier_model.parameters()):,}")
    print(f"clusterer_model模型结构:")
    # print(clusterer_model)
    print(f"总参数量: {sum(p.numel() for p in clusterer_model.parameters()):,}")
    
    # 创建示例数据
    x = torch.randn(batch_size, input_dim)
    # c = torch.randn(batch_size, condition_dim) # 有条件模型示例
    c = torch.zeros(batch_size, condition_dim)  # 无条件模型示例
    # 条件向量 非零：使用了类别标签作为条件；引入上下文信息（如时间、位置等）
    #         零向量：无条件模型，仅依赖输入数据本身进行建模。
    
    # 前向传播
    z1 = classifier_model.forward(x, c)
    z2 = clusterer_model.forward(x, c)
    print(f"\n前向传播:")
    print(f"输入 x 形状: {x.shape}")
    print(f"隐变量 z1 形状: {z1.shape}")
    print(f"\n对数似然形状: {z1.shape}")
    print(f"平均对数似然: {z1.mean().item():.4f}")

    print(f"\n前向传播:")
    print(f"输入 x 形状: {x.shape}")
    print(f"隐变量 z2 形状: {z2.shape}")
    print(f"\n对数似然形状: {z2.shape}")
    print(f"平均对数似然: {z2.mean().item():.4f}")