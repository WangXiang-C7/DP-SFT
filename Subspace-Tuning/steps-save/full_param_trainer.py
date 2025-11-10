import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  
from transformers import get_scheduler  
import os
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

class FullParamTrainer:
    """全参数微调训练器，用于第一阶段训练并保存参数轨迹"""
    def __init__(self, config, train_dataset, eval_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
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
            f"{config.dataset_name}_full_param"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存参数形状和大小
        self.param_shapes = {}
        self.param_sizes = {}
        
        # 创建轨迹保存目录（与文档1一致）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_dir = os.path.join(
            config.output_dir,
            "trajectory",
            f"{config.model_name.replace('/', '-')}_{config.dataset_name}_{timestamp}"
        )
        os.makedirs(self.trajectory_dir, exist_ok=True)
        
        # 记录存储配置（与文档1一致）
        with open(os.path.join(self.trajectory_dir, "config.txt"), "w") as f:
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Batch size: {config.batch_size}\n")
            f.write(f"Epochs: {config.full_param_epochs}\n")
            f.write(f"Learning rate: {config.full_param_lr}\n")
            f.write(f"Subspace dim: {config.subspace_dim}\n")
        
        # 获取日志记录器
        self.logger = config.logger
        self.logger.info(f"[FULL_PARAM] 轨迹保存目录: {self.trajectory_dir}")
        
        # 初始化轨迹索引和路径列表
        self.trajectory_index = 0
        self.trajectory_paths = []
    
    def save_model_params(self, model, step):
        """保存参数快照（与文档1完全一致）"""
        # 创建文件名（使用递增索引）
        file_path = os.path.join(self.trajectory_dir, f"{self.trajectory_index}.pt")
        self.trajectory_index += 1
        
        # 收集参数
        params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 保存参数原始形状（不展平）
                params_dict[name] = param.data.cpu()
        
        # === 保存参数形状到文件 ===
        param_shapes_path = os.path.join(self.trajectory_dir, "param_shapes.pt")
        torch.save(self.param_shapes, param_shapes_path)
        self.logger.info(f"[FULL_PARAM] 参数形状已保存: {param_shapes_path}")
        
        # 保存到磁盘
        torch.save({
            "step": step,             # 当前训练步数
            "params": params_dict,     # 参数字典（保持原始形状）
            "timestamp": time.time()   # 保存时间戳
        }, file_path)
        
        # 记录路径
        self.trajectory_paths.append(file_path)
        return file_path
    
    def train_with_trajectory(self, model, trajectory_save_interval):
        """训练模型并保存参数轨迹"""
        device = torch.device(self.config.device)
        model.to(device)
        
        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.full_param_lr
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_dataloader) * self.config.full_param_epochs
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_training_steps//10,
            num_training_steps=num_training_steps
        )
        
        # 在训练开始前记录参数形状（与文档1一致）
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_shapes[name] = param.data.shape
                self.param_sizes[name] = param.data.numel()
        
        # 记录初始参数（第0步）
        self.save_model_params(model, 0)
        self.logger.info(f"[FULL_PARAM] 保存初始参数轨迹")
        
        # 训练循环
        best_eval_metric = 0.0
        best_model_state = None
        
        # 创建进度条
        total_steps = num_training_steps
        progress_bar = tqdm(
            total=total_steps,
            desc=f"全参数训练 (Epoch 1/{self.config.full_param_epochs})",
            disable=not self.config.progress_bar,
            dynamic_ncols=True,
            miniters=self.config.log_interval,
            smoothing=0.05,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            leave=False  # 进度条完成后不保留
        )
        
        self.logger.info(f"\n[FULL_PARAM] 开始全参数微调...")
        self.logger.info(f"[FULL_PARAM] 训练轮次: {self.config.full_param_epochs}")
        self.logger.info(f"[FULL_PARAM] 总步数: {num_training_steps}")
        self.logger.info(f"[FULL_PARAM] 学习率: {self.config.full_param_lr}")
        self.logger.info(f"[FULL_PARAM] 批次大小: {self.config.batch_size}")
        self.logger.info(f"[FULL_PARAM] 轨迹保存间隔: 每 {trajectory_save_interval} 步保存一次")
        
        global_step = 0
        last_saved_step = -1
        for epoch in range(self.config.full_param_epochs):
            # 更新进度条描述
            progress_bar.set_description(
                f"全参数训练 (Epoch {epoch+1}/{self.config.full_param_epochs})"
            )
            
            self.logger.info(f"\n[FULL_PARAM] Epoch {epoch+1}/{self.config.full_param_epochs}")
            model.train()
            total_loss = 0.0
            losses = []
            start_time = time.time()
            
            for step, batch in enumerate(self.train_dataloader):
                # 准备数据
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                losses.append(loss.item())
                global_step += 1
                
                # 按step间隔保存轨迹（与文档1一致）
                if global_step % trajectory_save_interval == 0:
                    # 避免重复保存
                    if global_step != last_saved_step:
                        self.save_model_params(model, global_step)
                        self.logger.info(
                            f"[FULL_PARAM] 保存step {global_step}参数轨迹"
                        )
                        last_saved_step = global_step
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.8f}"
                })
                
                # 打印训练进度
                if (step + 1) % self.config.log_interval == 0:
                    avg_loss = np.mean(losses[-self.config.log_interval:])
                    self.logger.debug(
                        f"  [FULL_PARAM] Step {global_step}/{num_training_steps}, "
                        f"Loss: {avg_loss:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.8f}"
                    )
            
            # 计算平均损失
            avg_loss = total_loss / len(self.train_dataloader)
            epoch_time = time.time() - start_time
            self.logger.info(f"[FULL_PARAM] Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}秒")
            
            # 评估
            eval_metric = self.evaluate(model)
            self.logger.info(
                f"[FULL_PARAM] Epoch {epoch+1}, 评估指标: {eval_metric:.4f}"
            )
            
            # 保存最佳模型指标
            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                best_model_state = model.state_dict().copy()
                self.logger.info(
                    f"[FULL_PARAM] 找到更好的模型，评估指标: {best_eval_metric:.4f}"
                )
        
        # 关闭进度条
        progress_bar.close()
        
        # 保存最终参数（最后一步）
        self.save_model_params(model, global_step)
        self.logger.info(f"[FULL_PARAM] 保存最终参数轨迹 (step {global_step})")
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.logger.info(
                f"[FULL_PARAM] 全参数微调完成，最佳评估指标: {best_eval_metric:.4f}"
            )
        else:
            self.logger.info("[FULL_PARAM] 全参数微调完成，但未找到最佳模型")
        
        # 保存轨迹索引文件（与文档1一致）
        index_path = os.path.join(self.trajectory_dir, "index.pt")
        torch.save(self.trajectory_paths, index_path)
        self.logger.info(f"[FULL_PARAM] 轨迹索引已保存: {index_path}")
        self.logger.info(f"[FULL_PARAM] 共保存 {len(self.trajectory_paths)} 个轨迹快照")
        
        # 返回结果（与文档1不同）
        # 返回轨迹矩阵和初始展平参数（文档1不返回这些）
        trajectory_list = []
        for file_path in self.trajectory_paths:
            data = torch.load(file_path)
            # 展平所有参数
            flat_params = torch.cat([param.flatten() for param in data["params"].values()])
            trajectory_list.append(flat_params)
        
        trajectory_matrix = torch.stack(trajectory_list)
        initial_flat = trajectory_list[0]
        
        self.logger.info(f"[FULL_PARAM] 重建轨迹矩阵，形状: {trajectory_matrix.shape}")
        
        return model, trajectory_matrix, initial_flat, self.param_shapes
    
    def evaluate(self, model):
        """评估模型性能"""
        device = torch.device(self.config.device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        self.logger.info("[FULL_PARAM] 开始评估...")
        
        # 创建评估进度条
        eval_progress = tqdm(
            total=len(self.eval_dataloader),
            desc="评估进度",
            disable=not self.config.progress_bar,
            dynamic_ncols=True,
            leave=False  # 进度条完成后不保留
        )
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                
                eval_progress.update(1)
        
        eval_progress.close()
        
        # 计算准确率
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        return accuracy