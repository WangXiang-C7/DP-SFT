import torch
import evaluate
import os
import time
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from datetime import datetime
from dpsgd import PrivacyTracker, process_gradients, update_model

class FullParamTrainer:
    """全参数微调训练器（DP版本），用于第一阶段训练并保存参数轨迹"""
    def __init__(self, config, train_loader, eval_loader, loss_fn):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.loss_fn = loss_fn
        # 创建输出目录
        self.output_dir = os.path.join(
            config.output_dir,
            f"{config.dataset_name}_full_param"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存参数形状和大小
        self.param_shapes = {}
        self.param_sizes = {}
        
        # 获取日志记录器
        self.logger = config.logger
        
        # 计算总步数
        self.total_steps = config.stage1_epochs * (len(train_loader.dataset) // config.batch_size)
        
        # 计算轨迹保存间隔
        self.trajectory_save_interval = self.total_steps // config.stage2_subspace_dim
        
        # 初始化评估指标
        if config.dataset_name in ["sst2", "mnli", "qnli", "qqp", "mrpc", "rte", "wnli", "cola", "stsb"]:
            self.metric = evaluate.load(
                'glue', config.dataset_name, 
                experiment_id=f"fullparam_{config.dataset_name}"
            )
        elif config.dataset_name == "imdb":
            self.metric = evaluate.load(
                'accuracy',
                experiment_id=f"fullparam_{config.dataset_name}"
            )
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trajectory_dir = os.path.join(
            "trajectory",
            f"{config.model_name.replace('/', '-')}_{config.dataset_name}_{timestamp}"
        )
        os.makedirs(self.trajectory_dir, exist_ok=True)
        config.logger.info(f"[FULL_PARAM] 轨迹存储目录: {self.trajectory_dir}")
        
        # 记录存储配置
        with open(os.path.join(self.trajectory_dir, "config.txt"), "w") as f:
            f.write(f"Model: {config.model_name}\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Batch size: {config.batch_size}\n")
            f.write(f"Total steps: {self.total_steps}\n")
            f.write(f"Save interval: {self.trajectory_save_interval}\n")
        
        # 初始化轨迹索引
        self.trajectory_index = 0
        self.trajectory_paths = []
    
    def save_model_params(self, model, step):
        """保存参数快照到磁盘（使用顺序编号）"""
        # 创建文件名
        file_path = os.path.join(self.trajectory_dir, f"{self.trajectory_index}.pt")
        self.trajectory_index += 1
        
        # 收集参数
        params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_dict[name] = param.data.cpu()
        
        # 保存到磁盘
        torch.save({
            "step": step,
            "params": params_dict,
            "timestamp": time.time()
        }, file_path)
        
        # 记录路径
        self.trajectory_paths.append(file_path)
        return file_path
    
    def train_with_trajectory(self, model):
        """训练模型并保存参数轨迹到磁盘"""
        device = torch.device(self.config.device)
        model.to(device)
        
        # 优化器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.stage1_lr,
            weight_decay=self.config.stage1_weight_decay
        )
        
        # 学习率调度器
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(self.total_steps * self.config.stage1_warmup_ratio),
            num_training_steps=self.total_steps
        )
        
        # 初始化隐私跟踪器
        privacy_tracker = PrivacyTracker(self.config.stage1_target_delta)
        
        # 损失函数
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 训练循环
        self.logger.info(f"\n[FULL_PARAM] 开始全参数微调 (DP)...")
        self.logger.info(f"[FULL_PARAM] 训练总步数: {self.total_steps}")
        self.logger.info(f"[FULL_PARAM] 学习率: {self.config.stage1_lr}")
        self.logger.info(f"[FULL_PARAM] 批次大小: {self.config.batch_size}")
        self.logger.info(f"[FULL_PARAM] 梯度裁剪范数: {self.config.stage1_clip_norm}")
        self.logger.info(f"[FULL_PARAM] 噪声乘数: {self.config.stage1_noise_multiplier}")
        self.logger.info(f"[FULL_PARAM] 轨迹保存间隔: 每 {self.trajectory_save_interval} 步保存一次")
        self.logger.info(f"[FULL_PARAM] 轨迹存储目录: {self.trajectory_dir}")
        
        # 创建进度条
        progress_bar = tqdm(
            total=self.total_steps,
            desc="全参数训练 (DP)",
            disable=not self.config.progress_bar,
            dynamic_ncols=True,
            miniters=self.config.log_interval,
            smoothing=0.05,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            leave=True
        )
        
        # 训练主循环（基于步数）
        global_step = 0
        total_loss = 0.0
        losses = []
        start_time = time.time()
        
        # 在训练开始前记录参数形状
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_shapes[name] = param.data.shape
                self.param_sizes[name] = param.data.numel()
        
        # 记录初始参数（第0步）
        self.save_model_params(model, 0)
        
        model.train()
        while global_step < self.total_steps:
            # 获取当前批次
            try:
                batch = self.train_loader.get_batch(global_step)
                batch = {k: v.to(device) for k, v in batch.items()}
            except Exception as e:
                self.logger.error(f"获取批次时出错: {str(e)}")
                raise
            
            # DP模式下的梯度处理
            processed_grads = process_gradients(
                model, batch, loss_fn, 
                self.config.stage1_clip_norm,
                self.config.stage1_noise_multiplier,
                self.config.batch_size
            )
            
            # 更新模型参数
            update_model(model, processed_grads, optimizer)
            
            # 更新隐私跟踪器
            privacy_tracker.step(
                self.config.stage1_noise_multiplier,
                self.config.batch_size / len(self.train_loader.dataset)
            )
            
            # 获取损失值用于记录
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.item()
            
            # 更新学习率
            scheduler.step()
            
            # 记录损失
            total_loss += loss
            losses.append(loss)
            
            # 按step间隔保存轨迹
            if (global_step + 1) % self.trajectory_save_interval == 0:
                self.save_model_params(model, global_step + 1)
                self.logger.info(
                    f"[FULL_PARAM] 保存 step {global_step+1} 参数轨迹"
                )
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "mem": f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
            })
            
            # 定期内存监控
            if (global_step + 1) % 50 == 0:
                allocated = torch.cuda.memory_allocated()/1e9
                cached = torch.cuda.memory_reserved()/1e9
                self.logger.debug(
                    f"Step {global_step+1}: GPU内存 - 已分配: {allocated:.2f}GB, 保留: {cached:.2f}GB"
                )
            
            global_step += 1
        
        # 关闭进度条
        progress_bar.close()
        
        # 保存最终参数（最后一步）
        self.save_model_params(model, self.total_steps)
        
        # 计算总训练时间
        total_time = time.time() - start_time
        avg_loss = total_loss / self.total_steps
        
        # 最终评估
        self.logger.info(f"[FULL_PARAM] 训练完成，耗时: {total_time:.2f}秒")
        self.logger.info(f"[FULL_PARAM] 平均损失: {avg_loss:.4f}")
        
        # 隐私报告
        privacy_spent = privacy_tracker.get_privacy_spent()
        self.logger.info(
            f"[FULL_PARAM] 最终隐私消耗: ε={privacy_spent[0]:.2f}, δ={self.config.stage1_target_delta}"
        )
        
        # 最终评估
        eval_metric = self.evaluate_model(model)
        self.logger.info(
            f"[FULL_PARAM] 最终评估指标: {eval_metric:.4f}"
        )
        
        # 保存轨迹索引文件
        torch.save(self.trajectory_paths, os.path.join(self.trajectory_dir, "index.pt"))
        self.logger.info(
            f"[FULL_PARAM] 轨迹索引已保存: {os.path.join(self.trajectory_dir, 'index.pt')}"
        )
        self.logger.info(
            f"[FULL_PARAM] 共保存 {len(self.trajectory_paths)} 个轨迹快照"
        )
        
        return model, self.trajectory_dir, self.param_shapes
    
    def evaluate_model(self, model):
        """使用Hugging Face evaluate库评估模型性能"""
        device = torch.device(self.config.device)
        model.eval()
        self.logger.info("[FULL_PARAM] 开始最终评估...")
        
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
                if self.config.dataset_name == "stsb":
                    predictions = outputs.logits.squeeze()
                    # 使用MSE和Pearson相关性
                    mse = ((predictions - batch["labels"])**2).mean()
                    pearson = torch.corrcoef(torch.stack([predictions, batch["labels"]]))[0, 1]
                    self.metric.add_batch(predictions=predictions.cpu(), references=batch["labels"].cpu())
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