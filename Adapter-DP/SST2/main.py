import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dpsgd import process_gradients, update_model, PrivacyTracker, calculate_sigma
from loader import DPSGDDataLoader
from adapters import AutoAdapterModel, AdapterConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

class DPAdapterTrainer:
    """DP Adapter训练器（带实时损失监控和评估）"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和加载器
        self.model, self.tokenizer = self._init_model()
        
        # 训练数据加载器
        self.train_loader = DPSGDDataLoader(
            max_length=config.max_length,
            batch_size=config.batch_size,
            seed=config.seed,
            cache_dir=config.cache_dir
        )
        
        # 验证数据加载器
        self.val_loader = self._create_val_loader()
        
        # 计算总步数
        self.total_steps = config.epochs * (self.train_loader.dataset_size // config.batch_size)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        
        # DP设置
        if config.dp_enabled:
            self.sigma = calculate_sigma(
                target_epsilon=config.target_epsilon,
                target_delta=config.target_delta,
                sample_rate=config.batch_size / self.train_loader.dataset_size,
                total_steps=self.total_steps
            )
            self.privacy_tracker = PrivacyTracker(delta=config.target_delta)
            print(f"DP参数: ε={config.target_epsilon}, δ={config.target_delta}, σ={self.sigma:.4f}")
        else:
            self.sigma = None
            self.privacy_tracker = None
            
        # 跟踪最佳性能
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
    
    def _init_model(self):
        """初始化Adapter模型"""
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # 创建Adapter模型（内置分类头）
        model = AutoAdapterModel.from_pretrained(
            self.config.model_name,
            num_labels=2  # 二分类任务
        )
        model.add_classification_head(
            "sst2_head",
            num_labels=2,
            layers=2,  # 使用2层MLP
            activation_function='tanh',
            overwrite_ok=True,  # 允许覆盖
            multilabel=False,
            id2label=None,
            use_pooler=False  # RoBERTa不使用pooler输出
        )
        
        # 配置Adapter
        adapter_config = AdapterConfig.load(
            "houlsby", 
            reduction_factor=self.config.reduction_factor,
            non_linearity="relu",
            ln_before=True,
            ln_after=False
        )
        
        # 添加并激活Adapter
        model.add_adapter("task_adapter", config=adapter_config)
        model.train_adapter("task_adapter")
        model.set_active_adapters("task_adapter")
        
        # 检查参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        return model.to(self.device), tokenizer
    
    def _create_val_loader(self):
        """创建验证集数据加载器"""
        # 加载验证集
        val_dataset = load_dataset("glue", "sst2", split="validation", cache_dir=self.config.cache_dir)
        
        # 预处理验证集
        def preprocess_val(examples):
            tokenized = self.tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
                return_tensors=None
            )
            tokenized["labels"] = examples["label"]
            return tokenized
        
        tokenized_val = val_dataset.map(
            preprocess_val,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # 创建数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 创建DataLoader
        return DataLoader(
            tokenized_val,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
    
    def evaluate(self):
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # 计算损失
                loss = self.loss_fn(logits, batch["labels"])
                total_loss += loss.item() * batch["labels"].size(0)
                
                # 计算准确率
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == batch["labels"]).sum().item()
                total_correct += correct
                total_samples += batch["labels"].size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
    
    def _adapter_process_gradients(self, batch):
        """DP梯度处理"""
        device = self.device
        batch = {k: v.to(device) for k, v in batch.items()}
        actual_batch_size = batch['input_ids'].shape[0]
        
        # 初始化梯度累加器
        summed_gradients = {name: torch.zeros_like(param) 
                            for name, param in self.model.named_parameters() 
                            if param.requires_grad}
        
        total_norms = torch.zeros(actual_batch_size, device=device)
        
        # 逐样本处理
        for i in range(actual_batch_size):
            # 高效提取样本（减少内存拷贝）
            micro_batch = {
                k: v[i:i+1].detach().requires_grad_(False) 
                for k, v in batch.items()
            }
            
            self.model.zero_grad()
            outputs = self.model(**micro_batch)
            
            # 正确获取logits
            logits = outputs.logits
            
            loss = self.loss_fn(logits, micro_batch["labels"])
            loss.backward()
            
            # 计算梯度范数
            sample_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach().clone()
                    flat_grad = grad.view(-1)
                    sample_norm += torch.sum(flat_grad**2)
            
            sample_norm = torch.sqrt(sample_norm)
            total_norms[i] = sample_norm
            
            # 裁剪系数
            clip_coef = torch.clamp(self.config.clip_norm / (sample_norm + 1e-8), max=1.0)
            
            # 应用裁剪并累加
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    clipped_grad = param.grad * clip_coef
                    summed_gradients[name] += clipped_grad.detach()
            
            # 显存清理
            torch.cuda.empty_cache()
        
        # 添加噪声（在聚合后统一添加）
        noisy_gradients = {}
        for name, grad_sum in summed_gradients.items():
            std = self.config.clip_norm * self.sigma
            noise = torch.randn_like(grad_sum) * std
            noisy_gradients[name] = grad_sum + noise
        
        # 梯度平均
        processed_grads = {
            name: grad / actual_batch_size
            for name, grad in noisy_gradients.items()
        }
        
        self.model.zero_grad()
        return processed_grads
    
    def train(self):
        """训练主循环（带实时损失监控和评估）"""
        print(f"开始训练，总步数: {self.total_steps}")
        
        # 初始化进度条（添加loss显示）
        progress_bar = tqdm(
            total=self.total_steps, 
            desc="DP训练进度", 
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}"
        )
        
        # 初始化记录
        loss_history = []
        val_loss_history = []
        val_accuracy_history = []
        smoothing_factor = 0.1  # 指数平滑因子
        eval_interval = self.total_steps // config.epochs
        
        for step in range(self.total_steps):
            # 获取当前step的随机批次
            batch = self.train_loader.get_batch(step)
            
            if self.config.dp_enabled:
                # DP梯度处理
                processed_grads = self._adapter_process_gradients(batch)
                
                # 更新模型
                update_model(self.model, processed_grads, self.optimizer)
                
                # 更新隐私会计
                self.privacy_tracker.step(
                    self.sigma,
                    self.config.batch_size / self.train_loader.dataset_size
                )
                
                # 计算当前损失
                with torch.no_grad():
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    loss = self.loss_fn(logits, batch["labels"]).item()
            else:
                # 普通训练
                self.optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                loss = self.loss_fn(logits, batch["labels"])
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
            
            # 记录损失
            loss_history.append(loss)
            
            # 计算指数移动平均损失（平滑显示）
            if len(loss_history) == 1:
                smoothed_loss = loss_history[0]
            else:
                smoothed_loss = smoothing_factor * loss + (1 - smoothing_factor) * loss_history[-2]
            
            # 更新进度条
            progress_bar.set_postfix({"loss":f"{smoothed_loss:.4f}"})
            progress_bar.update(1)
            
            # 定期评估模型
            if (step+1) % eval_interval == 0 or step == self.total_steps - 1:
                val_loss, val_accuracy = self.evaluate()
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_accuracy)
                
                # 打印评估结果
                print(f"\n步骤 {step}/{self.total_steps} - 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
                
                # 切换回训练模式
                self.model.train()
        
        # 关闭进度条
        progress_bar.close()
        
        # 最终隐私报告
        if self.config.dp_enabled:
            eps_spent, _ = self.privacy_tracker.get_privacy_spent()
            print(f"\n最终隐私消耗: ε={eps_spent:.4f}, δ={self.config.target_delta}")
        
        # 保存最终模型
        self.save_model()
        
        # 返回训练历史
        return {
            "train_loss": loss_history,
            "val_loss": val_loss_history,
            "val_accuracy": val_accuracy_history
        }
    
    def save_model(self, best=False):
        """保存Adapter模型"""
        if best:
            output_dir = "./best_dp_adapter" if self.config.dp_enabled else "./best_adapter"
        else:
            output_dir = "./dp_adapter_output" if self.config.dp_enabled else "./adapter_output"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_all_adapters(output_dir)
        self.model.save_all_heads(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存性能指标
        if best:
            with open(os.path.join(output_dir, "performance.txt"), "w") as f:
                f.write(f"验证损失: {self.best_val_loss:.4f}\n")
                f.write(f"验证准确率: {self.best_val_accuracy:.4f}\n")
        
        print(f"模型保存至: {output_dir}")

class Config:
    """训练配置"""
    def __init__(self):
        # 数据集配置
        self.dataset = "sst2"
        self.model_name = "roberta-base"
        
        # 训练参数
        self.epochs = 5
        self.batch_size = 32
        self.lr = 5e-4
        self.weight_decay = 0.01
        self.max_length = 128
        self.seed = 42
        self.cache_dir = "./cache"
        
        # Adapter配置
        self.reduction_factor = 48
        
        # DP参数
        self.dp_enabled = True
        self.target_epsilon = 4.0
        self.target_delta = 1e-5
        self.clip_norm = 10

if __name__ == "__main__":
    # 初始化配置和训练器
    config = Config()
    trainer = DPAdapterTrainer(config)
    # 开始训练并获取损失历史
    history = trainer.train()