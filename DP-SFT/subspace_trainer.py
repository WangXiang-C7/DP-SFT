import torch
import os
import time
import evaluate
import numpy as np
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from dpsgd import PrivacyTracker

class SubspaceDPTrainer:
    """子空间差分隐私训练器"""
    def __init__(self, config, train_loader, eval_loader, loss_fn):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.loss_fn = loss_fn
        self.logger = config.logger
        
        # 创建输出目录
        self.output_dir = os.path.join(
            config.output_dir,
            f"{config.dataset_name}_subspace_dim{config.stage2_subspace_dim}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化评估指标
        if config.dataset_name in ["sst2", "mnli", "qnli", "qqp", "mrpc", "rte", "wnli", "cola", "stsb"]:
            self.metric = evaluate.load(
                'glue', config.dataset_name,
                experiment_id=f"subspace_{config.dataset_name}"
                )
        elif config.dataset_name == "imdb":
            # 为IMDB数据集使用通用准确率指标
            self.metric = evaluate.load(
                'accuracy',
                experiment_id=f"subspace_{config.dataset_name}"
            )
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")
        
        # 创建调试日志文件
        debug_dir = os.path.join(self.output_dir, "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.debug_log_path = os.path.join(debug_dir, f"grad_debug_{timestamp}.txt")
        self.debug_log = open(self.debug_log_path, "w")
        # 修改日志标题行以包含四种状态
        self.debug_log.write("Step\tBeforeClip\tAfterClip\tBeforeNoise\tAfterNoise\n")
        self.debug_log.flush()
        
        self.logger.info(f"[SUBSPACE] 调试日志保存至: {self.debug_log_path}")
        self.logger.info("[SUBSPACE] 梯度监控状态: 1.裁减前 2.裁减后 3.加噪前 4.加噪后")
    
    def project_to_subspace(self, gradients_dict, V):
        """
        将高维梯度投影到低维子空间
        
        参数:
            gradients_dict: 参数字典（参数名 -> 梯度张量）
            V: 投影矩阵 [D, d]  D=原始维度, d=子空间维度
        
        返回:
            low_dim_grads: 低维梯度向量 [batch_size, d]
        """
        # 获取批次大小（所有参数的梯度张量第一维必须相同）
        batch_size = next(iter(gradients_dict.values())).shape[0]
        
        # 获取子空间维度
        d = V.shape[1]  # 子空间维度
        
        # 初始化低维梯度张量，形状为 [batch_size, d]
        low_dim_grads = torch.zeros(batch_size, d, device=V.device)
        
        # 初始化指针，用于跟踪当前处理的参数在投影矩阵中的起始位置
        start_idx = 0
        
        # 遍历模型的每个参数及其对应的梯度
        for name, grad in gradients_dict.items():
            # 将梯度张量展平为二维张量 [batch_size, num_params]
            flat_grad = grad.view(grad.shape[0], -1)
            
            # 获取当前参数的维度（展平后的参数数量）
            num_params = flat_grad.shape[1]
            
            # 从投影矩阵V中提取当前参数对应的子矩阵
            # 每个参数在V中对应一个子矩阵，大小为 [num_params, d]
            V_sub = V[start_idx:start_idx+num_params, :]
            
            # 执行投影操作：将展平的梯度与子矩阵相乘
            # [batch_size, num_params] x [num_params, d] -> [batch_size, d]
            # 并累加到低维梯度张量中
            low_dim_grads += torch.matmul(flat_grad, V_sub)
            
            # 更新指针位置，移动到投影矩阵中下一个参数对应的位置
            start_idx += num_params
        
        return low_dim_grads
    
    def project_to_full_space(self, low_dim_grads, V, param_shapes):
        """
        将低维梯度投影回高维空间
        
        参数:
            low_dim_grads: 低维梯度向量 [batch_size, d]
            V: 投影矩阵 [D, d]
            param_shapes: 参数字典（参数名 -> 形状）
        
        返回:
            full_dim_grads: 高维梯度字典（参数名 -> 梯度张量）
        """
        # 计算原始高维空间的维度
        full_dim = V.shape[0]  # D = 原始参数空间维度
        
        # 获取批次大小
        batch_size = low_dim_grads.shape[0]
        
        # 执行投影操作，将低维梯度映射回高维空间
        # [batch_size, d] x [d, D] -> [batch_size, D]
        high_dim_grads = torch.matmul(low_dim_grads, V.t())
        
        # 初始化结果字典，用于存储重构后的各参数梯度
        full_dim_grads = {}
        
        # 初始化指针，用于跟踪当前处理的参数在高维梯度中的起始位置
        start_idx = 0
        
        # 遍历每个参数的原始形状信息
        for name, shape in param_shapes.items():
            # 计算当前参数的元素数量
            num_params = torch.prod(torch.tensor(shape)).item()
            
            # 从高维梯度中提取当前参数对应的部分
            param_grad = high_dim_grads[:, start_idx:start_idx+num_params]
            
            # 将提取的梯度重塑为原始参数形状
            full_dim_grads[name] = param_grad.view(batch_size, *shape)
            
            # 更新指针位置，移动到高维梯度中下一个参数对应的位置
            start_idx += num_params
    
        return full_dim_grads
    
    def clip_gradients(self, gradients, clip_norm):
        """
        裁剪每个样本的梯度（适用于低维子空间梯度）
        
        参数:
            gradients: 低维梯度张量，形状为 [batch_size, d]
            clip_norm: 裁剪阈值
            
        返回:
            clipped_grads: 裁剪后的梯度
            avg_norm: 裁剪前的平均梯度范数
        """
        # 计算每个样本的L2范数
        norms = torch.norm(gradients, p=2, dim=1)  # [batch_size]
        avg_norm = norms.mean().item()
        
        # 计算裁剪系数
        clip_coef = torch.clamp(clip_norm / (norms + 1e-8), max=1.0)
        
        # 应用裁剪
        clipped_grads = gradients * clip_coef.unsqueeze(1)
        
        # 返回裁剪后的梯度和范数信息
        return clipped_grads, avg_norm
    
    def subspace_dp_step(self, model, batch, V, param_shapes, clip_norm, noise_multiplier, step):
        # 1. 获取批次大小
        batch_size = next(iter(batch.values())).shape[0]
        device = next(model.parameters()).device
        d = V.shape[1]  # 子空间维度
        
        # 2. 初始化低维梯度累加器
        low_dim_grads = torch.zeros(batch_size, d, device=device)
        
        # 3. 处理每个样本（微批次大小为1）
        for i in range(batch_size):
            # 提取单个样本
            micro_batch = {k: v[i].unsqueeze(0) for k, v in batch.items()}
            
            # 计算单个样本的梯度
            model.zero_grad()
            outputs = model(**micro_batch)
            loss = self.loss_fn(outputs.logits, micro_batch["labels"])
            loss.backward()
            
            # 收集当前样本的梯度
            per_sample_grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # 克隆梯度并断开计算图
                    per_sample_grads[name] = param.grad.detach().clone().unsqueeze(0)  # 添加批次维度
            
            # 投影到低维子空间
            low_dim_grad = self.project_to_subspace(per_sample_grads, V)
            
            # 累加到总梯度
            low_dim_grads[i] = low_dim_grad.squeeze(0)  # 移除批次维度
            
            # 清理显存
            del per_sample_grads
            torch.cuda.empty_cache()
        
        # 4. 记录状态1: 裁剪前的梯度范数 (投影后)
        before_clip_norm = torch.norm(low_dim_grads, dim=1).mean().item()
        
        # 5. 梯度裁剪
        clipped_grads, clip_norm_value = self.clip_gradients(low_dim_grads, clip_norm)
        
        # 6. 记录状态2: 裁剪后的梯度范数
        after_clip_norm = torch.norm(clipped_grads, dim=1).mean().item()
        
        # 7. 梯度聚合
        summed_grads = torch.sum(clipped_grads, dim=0)
        
        # 8. 记录状态3: 加噪前的梯度范数
        before_noise_norm = torch.norm(summed_grads).item()
        
        # 9. 添加噪声
        noise_std = clip_norm * noise_multiplier
        noise = torch.randn_like(summed_grads) * noise_std
        noisy_grads = summed_grads + noise
        
        # 10. 记录状态4: 加噪后的梯度范数
        after_noise_norm = torch.norm(noisy_grads).item()
        
        # 11. 梯度平均
        avg_low_dim_grads = noisy_grads / batch_size
        
        # 12. 投影回高维空间
        full_dim_grads = self.project_to_full_space(
            avg_low_dim_grads.unsqueeze(0),  # 添加批次维度
            V, 
            param_shapes
        )
        
        # 13. 记录所有调试信息
        self.debug_log.write(
            f"{step}\t"
            f"{before_clip_norm:.6f}\t"    # 裁减前
            f"{after_clip_norm:.6f}\t"     # 裁减后
            f"{before_noise_norm:.6f}\t"    # 加噪前
            f"{after_noise_norm:.6f}\n"    # 加噪后
        )
        self.debug_log.flush()
        
        # 定期打印监控信息
        if step % self.config.log_interval == 0:
            self.logger.info(
                f"  [GRAD MONITOR] Step {step}: "
                f"BeforeClip={before_clip_norm:.4f}, "
                f"AfterClip={after_clip_norm:.4f}, "
                f"BeforeNoise={before_noise_norm:.4f}, "
                f"AfterNoise={after_noise_norm:.4f}"
            )
        
        return full_dim_grads
    
    def train(self, model, V, param_shapes):
        """
        子空间DP训练主函数
        
        参数:
            model: 基础模型
            V: 投影矩阵 [D, d]
            param_shapes: 参数字典（参数名 -> 形状）
        """
        device = torch.device(self.config.device)
        model.to(device)
        V = V.to(device)
        
        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.stage2_lr
        )
        
        # 学习率调度器
        total_steps = self.config.stage2_epochs * (len(self.train_loader.dataset) // self.config.batch_size)
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.config.stage2_warmup_ratio),
            num_training_steps=total_steps
        )
        
        # 初始化隐私跟踪器
        privacy_tracker = PrivacyTracker(self.config.stage2_target_delta)
        
        self.logger.info(f"\n[SUBSPACE] 开始子空间DP训练...")
        self.logger.info(f"[SUBSPACE] 训练总步数: {total_steps}")
        self.logger.info(f"[SUBSPACE] 子空间维度: {V.shape[1]}")
        self.logger.info(f"[SUBSPACE] 学习率: {self.config.stage2_lr}")
        self.logger.info(f"[SUBSPACE] 批次大小: {self.config.batch_size}")
        self.logger.info(f"[SUBSPACE] 梯度裁剪范数: {self.config.stage2_clip_norm}")
        self.logger.info(f"[SUBSPACE] 噪声乘数: {self.config.stage2_noise_multiplier}")
        
        # 创建进度条
        progress_bar = tqdm(
            total=total_steps,
            desc="子空间训练 (DP)",
            disable=not self.config.progress_bar,
            dynamic_ncols=True,
            miniters=self.config.log_interval,
            smoothing=0.05,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            leave=True
        )
        
        # 训练主循环
        global_step = 0
        total_loss = 0.0
        losses = []
        start_time = time.time()
        
        model.train()
        while global_step < total_steps:
            # 获取当前批次
            batch = self.train_loader.get_batch(global_step)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 子空间DP梯度处理
            processed_grads = self.subspace_dp_step(
                model, 
                batch,
                V,
                param_shapes,
                self.config.stage2_clip_norm,
                self.config.stage2_noise_multiplier,
                global_step
            )
            
            # 更新隐私跟踪器
            privacy_tracker.step(
                self.config.stage2_noise_multiplier,
                self.config.batch_size / len(self.train_loader.dataset)
            )
            
            # 使用处理后的梯度更新模型
            for name, param in model.named_parameters():
                if name in processed_grads:
                    # 使用平均梯度而不是第一个样本
                    param.grad = processed_grads[name].mean(dim=0).clone()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新学习率
            scheduler.step()
            
            # 获取损失值用于记录
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.item()
            
            # 记录损失
            total_loss += loss
            losses.append(loss)
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}"
            })
            
            # 打印训练进度
            if (global_step + 1) % self.config.log_interval == 0:
                avg_loss = np.mean(losses[-self.config.log_interval:])
                self.logger.info(
                    f"  [SUBSPACE] Step {global_step}/{total_steps}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f}"
                )
            
            global_step += 1
        
        # 关闭进度条
        progress_bar.close()
        
        # 关闭调试日志
        self.debug_log.close()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        avg_loss = total_loss / total_steps
        
        # 最终评估
        self.logger.info(f"[SUBSPACE] 训练完成，耗时: {total_time:.2f}秒")
        self.logger.info(f"[SUBSPACE] 平均损失: {avg_loss:.4f}")
        
        # 隐私报告
        privacy_spent = privacy_tracker.get_privacy_spent()
        self.logger.info(
            f"[SUBSPACE] 最终隐私消耗: ε={privacy_spent[0]:.2f}, δ={self.config.stage2_target_delta}"
        )
        
        # 最终评估
        eval_metric = self.evaluate(model)
        self.logger.info(
            f"[SUBSPACE] 最终评估指标: {eval_metric:.4f}"
        )
        
        return model, eval_metric
    
    def evaluate(self, model):
        """使用Hugging Face evaluate库评估模型性能"""
        device = torch.device(self.config.device)
        model.eval()
        
        self.logger.info("[SUBSPACE] 开始最终评估...")
        
        # 创建评估进度条
        eval_progress = tqdm(
            total=len(self.eval_loader),
            desc="评估进度",
            disable=not self.config.progress_bar,
            dynamic_ncols=True,
            leave=False
        )
        
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                # 根据不同任务处理预测结果
                if self.config.dataset_name == "stsb":  # 回归任务
                    predictions = outputs.logits.squeeze()
                else:  # 分类任务
                    predictions = outputs.logits.argmax(dim=-1)
                
                # 更新评估指标
                self.metric.add_batch(
                    predictions=predictions.detach().cpu().numpy(),
                    references=batch["labels"].cpu().numpy()
                )
                eval_progress.update(1)
        
        eval_progress.close()
        
        # 计算最终指标
        eval_results = self.metric.compute()
        
        # 提取主要指标
        if self.config.dataset_name == "stsb":
            main_metric = eval_results.get("pearson", 0.0)
        elif self.config.dataset_name == "cola":
            main_metric = eval_results.get("matthews_correlation", 0.0)
        elif self.config.dataset_name in ["mrpc", "qqp"]:
            main_metric = eval_results.get("f1", 0.0)
        else:
            main_metric = eval_results.get("accuracy", 0.0)
        
        return main_metric