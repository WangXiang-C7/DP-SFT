import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import math
from torch import optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from loader import prepare_datasets
from train import set_seed, train_with_dpsgd

def main():
    # 配置参数
    set_seed(42)
    config = {
        "max_length": 128,            # 最大序列长度
        "batch_size": 32,             # 批次大小
        "learning_rate": 5e-4,        # 调整学习率
        "weight_decay": 0.01,         # 权重衰减
        "num_epochs": 5,              # 训练轮数
        "grad_norm": 0.5,             # 梯度裁剪阈值
        "target_epsilon": 4,          # 目标隐私预算ε
        "target_delta": 1e-5,         # 目标失败概率δ
        "warmup_ratio": 0.1,          # 学习率预热比例
        "cache_dir": "./cache",       # 缓存目录
        "output_dir": "./output",     # 输出目录
    }
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 确保输出目录存在
    os.makedirs(config["output_dir"], exist_ok=True)
 
    # 1. 准备数据集
    qnli_datasets, tokenizer, collator = prepare_datasets(
        max_length=config["max_length"],
        cache_dir=config["cache_dir"]
    ) 
    
    # 计算采样率
    dataset_size = len(qnli_datasets["train"])
    sample_rate = config["batch_size"] / dataset_size
     
    # 创建训练数据加载器
    train_loader = DPDataLoader(
        dataset=qnli_datasets["train"],
        sample_rate=sample_rate,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(42) 
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        qnli_datasets["validation"], 
        batch_size=config["batch_size"],
        collate_fn=collator,
        shuffle=False
    )
     
    # 计算总训练步数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config["num_epochs"]
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    # 计算噪声乘数
    noise_multiplier = get_noise_multiplier(
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        sample_rate= sample_rate,
        steps=total_steps,
        accountant="rdp",
    ) 
    print(f"Calculated noise multiplier: {noise_multiplier:.4f}")
    
    # 2. 加载模型   
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
        cache_dir=config["cache_dir"]
    ).to(device)
    
    # 3. 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # 4. 学习率调度器
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    model.train()
    
    # 5. 初始化隐私引擎
    privacy_engine = PrivacyEngine()
    
    # 将模型、优化器和数据加载器转换为隐私
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        max_grad_norm=config["grad_norm"],
        epochs=config["num_epochs"],
    )

    
    # 6. 训练模型
    print("Starting training...")
    history, best_acc, final_epsilon = train_with_dpsgd(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        privacy_engine=privacy_engine,
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        epochs=config["num_epochs"],
        output_dir=config["output_dir"]
    )
    
    # 7. 最终评估
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Final privacy consumption: ε={final_epsilon:.2f} (δ={config['target_delta']})")

if __name__ == "__main__":
    main()