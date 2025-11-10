import torch
import torch.func as F
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
import math
from opacus.accountants import RDPAccountant

def calculate_sigma(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    total_steps: int,
    alphas: list = None,
    epsilon_tolerance: float = 0.01,
) -> float:
    """
    使用Opacus内置库函数计算达到目标隐私预算所需的噪声乘数
    
    参数:
    - target_epsilon: 目标ε值
    - target_delta: 目标δ值
    - sample_rate: 采样率 (batch_size / dataset_size)
    - total_steps: 训练总步长
    - alphas: RDP阶数列表，默认使用Opacus推荐值
    - epsilon_tolerance: 噪声乘数的下界
    
    返回:
    - sigma: 噪声乘数
    """
    # 扩展α范围 [1.1, 2.0, 3.0, ..., 100.0]
    if alphas is None:
        alphas = [1.1] + list(np.linspace(2, 100, 50))
    
    # 计算噪声乘数
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        alphas=alphas,
        epsilon_tolerance=epsilon_tolerance,
    )
    return sigma

def update_model(model, processed_grads, optimizer):
    """
    使用处理后的梯度更新模型参数
    
    参数:
    - model: 模型
    - processed_grads: 处理后的梯度
    - optimizer: 优化器
    - lr: 当前学习率
    """
    for name, param in model.named_parameters():
        if name in processed_grads and param.requires_grad:
            param.grad = processed_grads[name]
    optimizer.step()
    optimizer.zero_grad()

class PrivacyTracker:
    def __init__(self, delta):
        self.accountant = RDPAccountant()
        self.delta = delta
        self.steps = 0
    
    def step(self, sigma, sample_rate):
        self.accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        self.steps += 1
    
    def get_privacy_spent(self):
        return self.accountant.get_privacy_spent(delta=self.delta)

def process_gradients(model, batch, loss_fn, clip_norm, noise_multiplier, batch_size):
    """完整的梯度处理流程（使用微批次）"""
    device = next(model.parameters()).device
    
    # 初始化聚合梯度（用于累加所有样本的裁剪后梯度）
    summed_gradients = {name: torch.zeros_like(param) 
                        for name, param in model.named_parameters() 
                        if param.requires_grad}
    
    # 用于计算每个样本的整体梯度范数
    total_norms = torch.zeros(batch_size, device=device)
    
    # 处理每个样本（微批次大小为1）
    for i in range(batch_size):
        # 1. 提取单个样本
        micro_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}
        
        # 2. 计算单个样本的梯度
        model.zero_grad()
        outputs = model(**micro_batch)
        loss = loss_fn(outputs.logits, micro_batch["labels"])
        loss.backward()
        
        # 3. 收集当前样本的梯度并计算范数
        sample_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach().clone()
                
                # 计算该参数的梯度范数贡献
                flat_grad = grad.view(-1)
                sample_norm += torch.sum(flat_grad**2)
                
                # 临时保存梯度
                param.grad_sample = grad  # 临时存储
        
        # 4. 计算当前样本的总体梯度范数
        sample_norm = torch.sqrt(sample_norm)
        total_norms[i] = sample_norm
        
        # 5. 计算裁剪系数并应用裁剪
        clip_coef = torch.clamp(clip_norm / (sample_norm + 1e-8), max=1.0)
        
        # 6. 累加裁剪后的梯度
        for name, param in model.named_parameters():
            if param.requires_grad and hasattr(param, 'grad_sample'):
                clipped_grad = param.grad_sample * clip_coef
                summed_gradients[name] += clipped_grad
                
                # 清理临时存储
                del param.grad_sample
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 7. 添加噪声
    noisy_gradients = {}
    for name, grad_sum in summed_gradients.items():
        std = clip_norm * noise_multiplier
        noise = torch.randn_like(grad_sum) * std
        noisy_gradients[name] = grad_sum + noise
    
    # 8. 梯度平均
    processed_grads = {
        name: grad / batch_size
        for name, grad in noisy_gradients.items()
    }
    
    # 9. 清理模型中的梯度（防止内存泄漏）
    model.zero_grad()
    
    return processed_grads