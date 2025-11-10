import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import os, copy
import time
from tqdm import tqdm
import numpy as np
from torch.func import functional_call


class StrictSubspaceModel(torch.nn.Module):
    def __init__(self, base_model, V, w0, param_shapes, config):
        super().__init__()
        device = torch.device(config.device)
        
        # 创建基础模型的深拷贝
        self.base_model = copy.deepcopy(base_model).to(device)
        
        # 冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 设置模型为评估模式
        self.base_model.eval()
        
        self.V = V.to(device)
        self.w0 = w0.to(device)
        self.config = config
        
        # 构建参数索引映射
        self.param_info = {}
        ptr = 0
        for name, param in self.base_model.named_parameters():
            shape = param.shape
            size = param.numel()
            self.param_info[name] = {
                'size': size,
                'shape': shape,
                'start': ptr,
                'end': ptr + size
            }
            ptr += size
        
        # 可训练的theta参数 - 使用高斯初始化
        init_std = 0.01
        self.theta = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.randn(config.subspace_dim, device=device) * init_std,
                requires_grad=True
            ) for _ in range(config.ensemble_size)
        ])
        # 使用函数式参数绑定
        self.param_names = [name for name, _ in self.base_model.named_parameters()]
    
    def compute_full_params(self, theta_i):
        """计算完整参数: w = w0 + V * theta_i"""
        return self.w0 + torch.matmul(self.V, theta_i)
    
    def forward_single(self, theta_i, inputs):
        # 计算完整参数向量
        w_full = self.compute_full_params(theta_i)
        
        # 将参数向量分割为各层参数
        params_dict = {}
        ptr = 0
        for name in self.param_names:
            shape = self.param_info[name]['shape']
            size = self.param_info[name]['size']
            param_data = w_full[ptr:ptr+size].view(shape)
            params_dict[name] = param_data
            ptr += size
        
        # 使用函数式调用保持计算图
        return torch.func.functional_call(
            self.base_model,
            params_dict,
            args=(inputs["input_ids"],),
            kwargs={"attention_mask": inputs["attention_mask"], "labels": inputs["labels"]}
        )
    
    def forward_ensemble(self, inputs):
        """集成预测：返回所有成员的logits"""
        all_logits = []
        for i in range(self.config.ensemble_size):
            out = self.forward_single(self.theta[i], inputs)
            all_logits.append(out.logits)
        return torch.stack(all_logits)  # [h, batch_size, num_labels]

class StrictSubspaceTrainer:
    """子空间训练器"""
    def __init__(self, config, train_dataset, eval_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.grad_history = []
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        self.eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size
        )
        
        # 创建输出目录
        self.output_dir = os.path.join(
            config.output_dir,
            f"{config.dataset_name}_strict_subspace_dim{config.subspace_dim}_h{config.ensemble_size}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_member(self, model, member_idx):
        """训练单个集成成员（添加学习率调度）"""
        self.config.logger.info(f"\n[STRICT SUBSPACE] 训练成员 {member_idx+1}/{self.config.ensemble_size}")
        
        # 创建优化器（仅优化当前θ）
        optimizer = AdamW([model.theta[member_idx]], lr=self.config.subspace_lr)
        
        # ===== 学习率调度器 =====
        num_training_steps = len(self.train_dataloader) * self.config.subspace_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 训练循环
        for epoch in range(self.config.subspace_epochs):
            total_loss = 0.0
            epoch_start = time.time()
            step = 0  # 添加步数计数器
            
            progress_desc = f"成员 {member_idx+1} | Epoch {epoch+1}/{self.config.subspace_epochs}"
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=progress_desc,
                disable=not self.config.progress_bar,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            
            for batch in progress_bar:
                step += 1
                # 准备输入
                model_inputs = {
                    "input_ids": batch["input_ids"].to(self.config.device),
                    "attention_mask": batch["attention_mask"].to(self.config.device),
                    "labels": batch["labels"].to(self.config.device)
                }
                
                # 重置梯度
                optimizer.zero_grad()
                
                try:
                    # 前向传播
                    outputs = model.forward_single(model.theta[member_idx], model_inputs)
                    loss = outputs.loss
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度分析
                    grad = model.theta[member_idx].grad
                    grad_norm = torch.norm(grad).item() if grad is not None else 0.0
                    
                    # 检查梯度
                    if grad is None:
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "grad": "None",
                            "warning": "梯度消失!"
                        })
                        continue
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_([model.theta[member_idx]], max_norm=1.0)
                    
                    # 优化步骤
                    optimizer.step()
                    total_loss += loss.item()

                    # ===== 学习率更新 =====
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # 梯度统计（每50步记录一次）
                    if step % 50 == 0:
                        grad_stats = {
                            "step": step,
                            "epoch": epoch,
                            "member": member_idx,
                            "grad_norm": grad_norm,
                            "grad_mean": grad.mean().item(),
                            "grad_std": grad.std().item(),
                            "grad_min": grad.min().item(),
                            "grad_max": grad.max().item(),
                            "loss": loss.item()
                        }
                        self.grad_history.append(grad_stats)
                    
                    # 梯度监控阈值
                    grad_status = "正常"
                    if grad_norm < 1e-8:  # 梯度消失警告
                        grad_status = "消失警告!"
                    elif grad_norm > 1.0:  # 梯度爆炸警告
                        grad_status = "爆炸警告!"
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "|g|": f"{grad_norm:.2e}",
                        "g_mean": f"{grad.mean().item():.2e}",
                        "g_std": f"{grad.std().item():.2e}",
                        "lr": f"{current_lr:.2e}",  
                        "status": grad_status
                    })
                
                except RuntimeError as e:
                    progress_bar.set_postfix({"error": str(e)[:20]})
                    self.config.logger.error(f"训练错误: {str(e)}")
                    self.config.logger.error("跳过当前批次...")
                    continue
            
            # 计算平均损失
            avg_loss = total_loss / len(self.train_dataloader)
            epoch_time = time.time() - epoch_start
            msg1 = f"[STRICT SUBSPACE] 成员 {member_idx+1} | Epoch {epoch+1} | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.2f}秒"
            self.config.logger.info(msg1)
            if self.config.progress_bar:
                progress_bar.write(msg1)
            else:
                print(msg1)
            
            # 打印本epoch的梯度摘要
            epoch_grads = [g for g in self.grad_history if g["epoch"] == epoch and g["member"] == member_idx]
            if epoch_grads:
                avg_grad_norm = sum(g["grad_norm"] for g in epoch_grads) / len(epoch_grads)
                min_grad = min(g['grad_norm'] for g in epoch_grads)
                max_grad = max(g['grad_norm'] for g in epoch_grads)
                msg2 = f"梯度摘要: 平均范数={avg_grad_norm:.2e}, 最小={min_grad:.2e}, 最大={max_grad:.2e}"
                self.config.logger.info(msg2)
                if self.config.progress_bar:
                    progress_bar.write(msg2)
                else:
                    print(msg2)
    
    def evaluate_ensemble(self, model):
        """评估集成模型性能"""
        model.eval()
        device = self.config.device
        total_correct = 0
        total_samples = 0
        
        self.config.logger.info("[STRICT SUBSPACE] 评估集成模型...")
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="评估"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 获取所有成员的预测
                all_logits = model.forward_ensemble(batch)  # [h, batch_size, num_labels]
                
                # 集成平均
                avg_logits = torch.mean(all_logits, dim=0)
                predictions = avg_logits.argmax(dim=-1)
                
                # 计算准确率
                correct = (predictions == batch["labels"]).sum().item()
                total_correct += correct
                total_samples += batch["labels"].size(0)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def train(self, base_model, V, w0, param_shapes):
        """训练所有集成成员"""
        device = torch.device(self.config.device)

        # 确保所有输入在正确设备上
        base_model = base_model.to(device)
        V = V.to(device)
        w0 = w0.to(device)
        
        # 创建严格子空间模型
        model = StrictSubspaceModel(
            base_model.to(device),
            V.to(device),
            w0.to(device),
            param_shapes,
            self.config
        )
        
        # 存储每个成员的性能
        member_performance = []
        
        # 训练每个成员
        for i in range(self.config.ensemble_size):
            self.train_member(model, i)
            
            # 保存当前成员的θ
            if self.config.save_member_params:
                torch.save(
                    model.theta[i].detach().cpu(),
                    os.path.join(self.output_dir, f"theta_member_{i+1}.pt")
                )
                self.config.logger.info(f"[STRICT SUBSPACE] 保存成员 {i+1} 的 θ 参数")
            
            # 评估当前集成（包括新成员）
            accuracy = self.evaluate_ensemble(model)
            member_performance.append(accuracy)
            msg = f"[STRICT SUBSPACE] 成员 {i+1}加入后 | 集成准确率: {accuracy:.4f}"
            self.config.logger.info(msg)
            print(msg)
            
            # 保存中间模型
            torch.save(
                {
                    'theta': [model.theta[j].detach().cpu() for j in range(i+1)],
                    'performance': member_performance
                },
                os.path.join(self.output_dir, f"intermediate_model_{i+1}.pt")
            )
            self.config.logger.info(f"[STRICT SUBSPACE] 保存中间模型 {i+1}")
        
        # 最终评估
        final_accuracy = self.evaluate_ensemble(model)
        msg = f"[STRICT SUBSPACE] 所有成员训练完成 | 最终集成准确率: {final_accuracy:.4f}"
        self.config.logger.info(msg)
        print(msg)
        
        # 保存最终模型
        torch.save(
            {
                'theta': [theta_i.detach().cpu() for theta_i in model.theta],
                'performance': member_performance,
                'final_accuracy': final_accuracy
            },
            os.path.join(self.output_dir, "final_model.pt")
        )
        self.config.logger.info(f"[STRICT SUBSPACE] 保存最终模型")
        
        return final_accuracy