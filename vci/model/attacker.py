"""
Attacker Network Architecture for Model Inversion Attack

由于 Marson 数据集是 1D 向量（2000维）而不是 2D 图像，
我们不需要复杂的卷积神经网络 (CNN)，
多层感知机 (MLP) 是最高效、最直接的选择。

目标：学习映射 f_attack: Z → Ŷ
其中 Z 是从冻结 VCI encoder 提取的隐变量（128维）
Ŷ 是重构的基因表达向量（2000维）
"""

import torch
import torch.nn as nn
from typing import Optional


class AttackerMLP(nn.Module):
    """
    Multi-Layer Perceptron for Model Inversion Attack
    
    网络架构：
    - 输入层: Z (128维)
    - 隐藏层: 采用逐步放大的漏斗状结构（128 -> 512 -> 1024 -> 2000）
    - 输出层: 重构的 Ŷ (2000维)
    
    关键设计：
    - 激活函数：ReLU/LeakyReLU 在隐藏层间
    - 正则化：轻量级 Dropout 防止过拟合
    - 输出激活：根据数据预处理方式选择
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 2000,
        hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_batch_norm: bool = False
    ):
        """
        初始化攻击者 MLP 网络
        
        Args:
            input_dim: 输入维度 (Z 的维度，通常为 128)
            output_dim: 输出维度 (Y 的维度，通常为 2000)
            hidden_dims: 隐藏层维度列表
                        默认: [512, 1024] 实现 128 -> 512 -> 1024 -> 2000 的放大
            dropout_rate: Dropout 比率 (默认 0.1-0.2 用于防止过拟合)
            activation: 隐藏层激活函数 ("relu" 或 "leaky_relu")
            output_activation: 输出层激活函数
                             - None (Identity): 数据做过标准化（StandardScaler）
                             - "sigmoid": 数据规一化到 [0, 1]
            use_batch_norm: 是否使用 Batch Normalization
        """
        super().__init__()
        
        # 默认隐藏层结构：逐步放大的漏斗
        if hidden_dims is None:
            hidden_dims = [512, 1024]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        layers = []
        
        # 第一个隐藏层：input_dim -> hidden_dims[0]
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(self._get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # 中间隐藏层：hidden_dims[i] -> hidden_dims[i+1]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # 最后一个隐藏层：hidden_dims[-1] -> output_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 输出激活函数（关键点）
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            layers.append(nn.ReLU())
        elif output_activation is None:
            # Identity: 直接输出，不加激活函数
            # 适用于数据标准化场景
            pass
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.network = nn.Sequential(*layers)
    
    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 输入隐变量 (batch_size, 128)
        
        Returns:
            y_hat: 重构的基因表达向量 (batch_size, 2000)
        """
        return self.network(z)
    
    def get_architecture_summary(self) -> str:
        """获取网络架构概览"""
        summary = f"""
AttackerMLP Architecture Summary:
{'='*60}
Input Dimension:        {self.input_dim}
Output Dimension:       {self.output_dim}
Hidden Dimensions:      {self.hidden_dims}
Full Architecture:      {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}
Activation Function:    {self.activation.upper()}
Output Activation:      {self.output_activation if self.output_activation else 'Identity (no activation)'}
Dropout Rate:           {self.dropout_rate}
Batch Normalization:    {self.use_batch_norm}
{'='*60}
Total Parameters:       {sum(p.numel() for p in self.parameters()):,}
Trainable Parameters:   {sum(p.numel() for p in self.parameters() if p.requires_grad):,}
        """
        return summary


def create_attacker_mlp(
    latent_dim: int = 128,
    gene_dim: int = 2000,
    architecture: str = "default",
    dropout_rate: float = 0.1,
    output_activation: Optional[str] = None
) -> AttackerMLP:
    """
    工厂函数：创建不同预设的攻击者 MLP
    
    Args:
        latent_dim: 潜在空间维度 (通常 128)
        gene_dim: 基因表达维度 (通常 2000)
        architecture: 网络架构预设
            - "default": [512, 1024] - 中等规模 (推荐)
            - "small": [256, 512] - 小规模
            - "large": [1024, 2048] - 大规模
            - "deep": [256, 512, 1024] - 深度架构
        dropout_rate: Dropout 比率
        output_activation: 输出层激活函数
    
    Returns:
        AttackerMLP 实例
    """
    
    architectures = {
        "small": [256, 512],
        "default": [512, 1024],
        "large": [1024, 2048],
        "deep": [256, 512, 1024],
    }
    
    if architecture not in architectures:
        raise ValueError(f"Architecture {architecture} not found. "
                        f"Choose from {list(architectures.keys())}")
    
    hidden_dims = architectures[architecture]
    
    model = AttackerMLP(
        input_dim=latent_dim,
        output_dim=gene_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation="relu",
        output_activation=output_activation,
        use_batch_norm=False
    )
    
    return model


if __name__ == "__main__":
    # 测试网络架构
    model = create_attacker_mlp(
        latent_dim=128,
        gene_dim=2000,
        architecture="default",
        dropout_rate=0.1,
        output_activation=None  # 或 "sigmoid" 取决于数据预处理
    )
    
    print(model.get_architecture_summary())
    
    # 测试前向传播
    z = torch.randn(32, 128)  # Batch of 32 samples
    y_hat = model(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {y_hat.shape}")
    print(f"Expected output shape: (32, 2000)")
