import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class ConditionalCouplingLayer(nn.Module):
    """
    条件可逆耦合层
    根据Ardizzone等人的架构实现
    """
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 split_idx: Optional[int] = None):
        """
        初始化条件耦合层
        
        参数:
        ----------
        input_dim : int
            输入向量的维度
        condition_dim : int
            条件向量的维度
        hidden_dims : List[int]
            子网络的隐藏层维度
        split_idx : Optional[int]
            拆分索引，如果为None，则使用中间拆分
        """
        super(ConditionalCouplingLayer, self).__init__()
        
        # 如果未指定拆分索引，则从中间拆分
        if split_idx is None:
            self.split_idx = input_dim // 2
        else:
            self.split_idx = split_idx
        
        # 计算拆分后两部分的维度
        self.d1 = self.split_idx
        self.d2 = input_dim - self.split_idx
        
        # 创建四个子网络：s1, t1, s2, t2
        # 每个子网络都是全连接神经网络
        
        # s1 和 t1 网络：以 x_j^{(2)} 和条件 c 为输入
        in_dim_1 = self.d2 + condition_dim
        
        # s2 和 t2 网络：以 x_{j+1}^{(1)} 和条件 c 为输入
        in_dim_2 = self.d1 + condition_dim
        
        # 构建s1网络
        self.s1_net = self._build_network(in_dim_1, self.d1, hidden_dims)
        
        # 构建t1网络
        self.t1_net = self._build_network(in_dim_1, self.d1, hidden_dims)
        
        # 构建s2网络
        self.s2_net = self._build_network(in_dim_2, self.d2, hidden_dims)
        
        # 构建t2网络
        self.t2_net = self._build_network(in_dim_2, self.d2, hidden_dims)
        
    def _build_network(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        """构建全连接子网络"""
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 可以根据需要替换为其他激活函数
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（从数据空间到隐变量空间）
        
        参数:
        ----------
        x : torch.Tensor
            输入向量，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (batch_size, condition_dim)
        compute_jacobian : bool
            是否计算雅可比行列式的对数
        
        返回:
        ----------
        z : torch.Tensor
            变换后的向量
        log_det_jacobian : Optional[torch.Tensor]
            雅可比行列式的对数，如果compute_jacobian=False则为None
        """
        batch_size = x.shape[0]
        
        # 将输入向量拆分为两部分
        x1 = x[:, :self.d1]  # x_j^{(1)}
        x2 = x[:, self.d1:]  # x_j^{(2)}
        
        # 计算第一部分的变换
        # 拼接 x2 和条件 c
        input_s1_t1 = torch.cat([x2, c], dim=1)
        
        # 计算 s1 和 t1
        s1 = self.s1_net(input_s1_t1)
        t1 = self.t1_net(input_s1_t1)
        
        # 应用变换：x_{j+1}^{(1)} = x_j^{(1)} ⊙ exp(s1) ⊕ t1
        z1 = x1 * torch.exp(s1) + t1
        
        # 计算第二部分的变换
        # 拼接 z1 和条件 c
        input_s2_t2 = torch.cat([z1, c], dim=1)
        
        # 计算 s2 和 t2
        s2 = self.s2_net(input_s2_t2)
        t2 = self.t2_net(input_s2_t2)
        
        # 应用变换：x_{j+1}^{(2)} = x_j^{(2)} ⊙ exp(s2) ⊕ t2
        z2 = x2 * torch.exp(s2) + t2
        
        # 合并结果
        z = torch.cat([z1, z2], dim=1)
        
        # 计算雅可比行列式的对数（如果要求）
        if compute_jacobian:
            # 雅可比行列式的对数 = sum(s1) + sum(s2)
            log_det_jacobian = torch.sum(s1, dim=1) + torch.sum(s2, dim=1)
            return z, log_det_jacobian
        else:
            return z, None
    
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        逆变换（从隐变量空间到数据空间）
        
        参数:
        ----------
        z : torch.Tensor
            隐变量，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (batch_size, condition_dim)
        
        返回:
        ----------
        x : torch.Tensor
            重建的输入向量
        """
        # 将隐变量拆分为两部分
        z1 = z[:, :self.d1]  # x_{j+1}^{(1)}
        z2 = z[:, self.d1:]  # x_{j+1}^{(2)}
        
        # 计算 s2 和 t2（使用 z1 和条件 c）
        input_s2_t2 = torch.cat([z1, c], dim=1)
        s2 = self.s2_net(input_s2_t2)
        t2 = self.t2_net(input_s2_t2)
        
        # 逆变换：x_j^{(2)} = (z2 - t2) ⊙ exp(-s2)
        x2 = (z2 - t2) * torch.exp(-s2)
        
        # 计算 s1 和 t1（使用 x2 和条件 c）
        input_s1_t1 = torch.cat([x2, c], dim=1)
        s1 = self.s1_net(input_s1_t1)
        t1 = self.t1_net(input_s1_t1)
        
        # 逆变换：x_j^{(1)} = (z1 - t1) ⊙ exp(-s1)
        x1 = (z1 - t1) * torch.exp(-s1)
        
        # 合并结果
        x = torch.cat([x1, x2], dim=1)
        
        return x


class ConditionalINN(nn.Module):
    """
    条件可逆神经网络（cINN）
    由多个条件耦合层组成
    """
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 num_coupling_layers: int = 12,
                 hidden_dims: List[int] = [256, 256],
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed'):
        """
        初始化cINN模型
        
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
        """
        super(ConditionalINN, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_coupling_layers = num_coupling_layers
        self.use_permutation = use_permutation
        
        # 创建耦合层列表
        self.coupling_layers = nn.ModuleList()
        
        # 如果需要置换，创建置换列表
        if use_permutation:
            self.permutations = nn.ModuleList() if permutation_type == 'learnable' else []
            self.permutation_type = permutation_type
        
        for i in range(num_coupling_layers):
            # 创建耦合层
            # 可以交替改变拆分点以增强表达能力
            split_idx = None
            if i % 2 == 0:
                split_idx = input_dim // 2
            else:
                split_idx = input_dim // 3  # 不同的拆分点
            
            coupling_layer = ConditionalCouplingLayer(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dims=hidden_dims,
                split_idx=split_idx
            )
            self.coupling_layers.append(coupling_layer)
            
            # 如果不是最后一层，添加置换操作
            if use_permutation and i < num_coupling_layers - 1:
                if permutation_type == 'learnable':
                    # 可学习1x1卷积置换
                    perm_layer = nn.Conv1d(1, 1, 1, bias=False)
                    # 初始化为单位矩阵
                    nn.init.eye_(perm_layer.weight)
                    self.permutations.append(perm_layer)
                # 对于固定置换，我们将在forward中实现
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：从数据空间到隐变量空间
        
        参数:
        ----------
        x : torch.Tensor
            输入数据，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (batch_size, condition_dim)
        compute_jacobian : bool
            是否计算雅可比行列式的对数
        
        返回:
        ----------
        z : torch.Tensor
            隐变量，形状为 (batch_size, input_dim)
        log_det_jacobian : Optional[torch.Tensor]
            总雅可比行列式的对数
        """
        batch_size = x.shape[0]
        z = x
        total_log_det = torch.zeros(batch_size, device=x.device) if compute_jacobian else None
        
        for i, coupling_layer in enumerate(self.coupling_layers):
            # 通过耦合层
            if compute_jacobian:
                z, log_det = coupling_layer(z, c, compute_jacobian=True)
                total_log_det = total_log_det + log_det
            else:
                z, _ = coupling_layer(z, c, compute_jacobian=False)
            
            # 如果不是最后一层，应用置换
            if self.use_permutation and i < len(self.coupling_layers) - 1:
                if self.permutation_type == 'learnable':
                    # 可学习置换
                    z_reshaped = z.unsqueeze(1)  # 添加通道维度
                    z_permuted = self.permutations[i](z_reshaped)
                    z = z_permuted.squeeze(1)
                else:
                    # 固定置换：反转顺序
                    z = torch.flip(z, dims=[1])
        
        return z, total_log_det
    
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        逆变换：从隐变量空间到数据空间
        
        参数:
        ----------
        z : torch.Tensor
            隐变量，形状为 (batch_size, input_dim)
        c : torch.Tensor
            条件向量，形状为 (batch_size, condition_dim)
        
        返回:
        ----------
        x : torch.Tensor
            重建的数据
        """
        x = z
        
        # 逆序处理耦合层
        for i in reversed(range(len(self.coupling_layers))):
            # 如果不是第一层，先应用逆置换
            if self.use_permutation and i > 0:
                if self.permutation_type == 'learnable':
                    # 可学习置换的逆
                    x_reshaped = x.unsqueeze(1)  # 添加通道维度
                    x_permuted = self.permutations[i-1](x_reshaped)
                    x = x_permuted.squeeze(1)
                else:
                    # 固定置换的逆：再次反转顺序
                    x = torch.flip(x, dims=[1])
            
            # 通过耦合层的逆变换
            coupling_layer = self.coupling_layers[i]
            x = coupling_layer.inverse(x, c)
        
        return x
    
    def sample(self, c: torch.Tensor, num_samples: int = 1, 
               prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        """
        从条件分布中采样
        
        参数:
        ----------
        c : torch.Tensor
            条件向量，形状为 (batch_size, condition_dim) 或 (condition_dim,)
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
            c = c.unsqueeze(0)  # 添加批次维度
        
        batch_size = c.shape[0]
        
        # 如果没有提供先验分布，使用标准正态分布
        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=c.device),
                torch.ones(self.input_dim, device=c.device)
            )
        
        # 从先验分布中采样
        if num_samples > 1:
            # 扩展条件向量以匹配采样数量
            c_expanded = c.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.condition_dim)
            # 采样隐变量
            z = prior_dist.sample((batch_size * num_samples,)).to(c.device)
            # 通过逆变换生成样本
            samples = self.inverse(z, c_expanded)
            # 重塑为 (batch_size, num_samples, input_dim)
            samples = samples.reshape(batch_size, num_samples, -1)
        else:
            # 采样隐变量
            z = prior_dist.sample((batch_size,)).to(c.device)
            # 通过逆变换生成样本
            samples = self.inverse(z, c)
        
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
            条件向量，形状为 (batch_size, condition_dim)
        prior_dist : Optional[torch.distributions.Distribution]
            先验分布，默认为标准正态分布
        
        返回:
        ----------
        log_prob : torch.Tensor
            数据的对数似然，形状为 (batch_size,)
        """
        # 如果没有提供先验分布，使用标准正态分布
        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=x.device),
                torch.ones(self.input_dim, device=x.device)
            )
        
        # 前向传播获取隐变量和雅可比行列式
        z, log_det_jacobian = self.forward(x, c, compute_jacobian=True)
        
        # 计算先验分布下的对数概率
        log_prob_prior = prior_dist.log_prob(z).sum(dim=1)
        
        # 通过变量变换公式：log p(x|c) = log p(z) + log|det(J)|
        log_prob = log_prob_prior + log_det_jacobian
        
        return log_prob


# 示例：使用cINN模型
if __name__ == "__main__":
    # 设置超参数
    input_dim = 10
    condition_dim = 5
    batch_size = 32
    
    # 创建模型
    model = ConditionalINN(
        input_dim=input_dim,
        condition_dim=condition_dim,
        num_coupling_layers=6,
        hidden_dims=[128, 128],
        use_permutation=True,
        permutation_type='fixed'
    )
    
    print(f"模型结构:")
    print(model)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建示例数据
    x = torch.randn(batch_size, input_dim)
    c = torch.randn(batch_size, condition_dim)
    
    # 前向传播
    z, log_det = model.forward(x, c, compute_jacobian=True)
    print(f"\n前向传播:")
    print(f"输入 x 形状: {x.shape}")
    print(f"隐变量 z 形状: {z.shape}")
    print(f"雅可比行列式对数形状: {log_det.shape}")
    
    # 逆变换
    x_reconstructed = model.inverse(z, c)
    reconstruction_error = torch.mean((x - x_reconstructed) ** 2)
    print(f"\n重建误差 (MSE): {reconstruction_error.item():.6f}")
    
    # 计算对数似然
    log_prob = model.log_prob(x, c)
    print(f"\n对数似然形状: {log_prob.shape}")
    print(f"平均对数似然: {log_prob.mean().item():.4f}")
    
    # 采样
    samples = model.sample(c, num_samples=3)
    print(f"\n采样数据形状: {samples.shape}")