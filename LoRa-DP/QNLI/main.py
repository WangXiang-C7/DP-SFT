import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from loader import prepare_datasets
from train import set_seed, train_model  

def count_trainable_parameters(model):
    """计算可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 配置参数
    set_seed(42)
    config = {
        "max_length": 128,            # 序列长度
        "batch_size": 32,             # 批次大小
        "learning_rate": 5e-4,        # 学习率
        "weight_decay": 0.01,         # 权重衰减
        "num_epochs": 5,              # 训练轮数
        "warmup_ratio": 0.1,          # 学习率预热比例
        "cache_dir": "./cache",       # 缓存目录
        "output_dir": "./output",     # 输出目录
        "lora_rank": 16,              # LoRA秩
        "lora_alpha": 32,             # LoRA缩放因子
        "lora_dropout": 0.05,         # LoRA Dropout
        "lora_target_modules": ["query", "value", "key", "dense"],  # LoRA目标模块
        # ===== DP配置 =====
        "target_epsilon": 4.0,        # 隐私预算
        "target_delta": 1e-5,         # δ值
        "max_per_sample_grad_norm": 10,  # 梯度裁剪阈值
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
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        qnli_datasets["train"], 
        batch_size=config["batch_size"],
        collate_fn=collator,
        shuffle=False
    )
    
    # 创建验证数据加载器（QNLI使用validation作为验证集）
    val_loader = DataLoader(
        qnli_datasets["validation"],
        batch_size=config["batch_size"],
        collate_fn=collator,
        shuffle=False
    )

    # 2. 加载模型并添加LoRA适配器  
    base_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,  # QNLI是二分类（蕴含/不蕴含）
        cache_dir=config["cache_dir"]
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    # 打印可训练参数信息
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)")
    
    # 3. 设置优化器（仅优化LoRA参数）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # ===== 4. 初始化差分隐私引擎 =====
    # 获取数据集大小
    train_size = len(qnli_datasets["train"])
    # 使用Opacus内置方法计算噪声乘子
    noise_multiplier = get_noise_multiplier(
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        sample_rate=config["batch_size"] / train_size,
        epochs=config["num_epochs"],
        accountant="rdp",
    )
    
    config["noise_multiplier"] = noise_multiplier
    print(f"Calculated noise multiplier: {noise_multiplier:.4f}")
    
    # 创建DP引擎
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=config["max_per_sample_grad_norm"],
        poisson_sampling=True,
    )
    model = model.to(device)
    
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 6. 训练模型
    print("Starting DP-LoRA fine-tuning on QNLI dataset...")
    history, final_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config["num_epochs"],
        output_dir=config["output_dir"],
    )
    
    # 7. 最终评估和隐私报告
    print("\nTraining completed!")
    print(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final validation F1: {final_metrics['f1']:.4f}")
    print(f"Privacy cost: ε={config['target_epsilon']}, δ={config['target_delta']}")
    
if __name__ == "__main__":
    main()