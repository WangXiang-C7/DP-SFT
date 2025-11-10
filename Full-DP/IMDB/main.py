import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import logging
from torch import optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from loader import prepare_datasets
from train import set_seed, train_with_dpsgd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("dp_imdb_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 设置随机种子
    set_seed(42)
    
    # IMDB专用配置参数
    config = {
        "max_length": 256,            # 最大序列长度
        "batch_size": 32,             # 批次大小
        "learning_rate": 5e-4,        # 学习率
        "weight_decay": 0.01,         # 权重衰减
        "num_epochs": 5,              # 最大训练轮数
        "grad_norm": 0.5,             # 梯度裁剪阈值
        "target_epsilon": 4.0,        # 目标隐私预算ε
        "target_delta": 1e-5,         # 目标松弛项δ
        "warmup_ratio": 0.1,          # 学习率预热比例
        "cache_dir": "./cache",       # 缓存目录
        "output_dir": "./output_imdb",# 输出目录
    }
    
    # 打印配置
    logger.info("Starting IMDB DP training with configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 确保输出目录存在
    os.makedirs(config["output_dir"], exist_ok=True)
 
    # 1. 准备IMDB数据集
    logger.info("Preparing IMDB dataset...")
    imdb_datasets, tokenizer, collator = prepare_datasets(
        max_length=config["max_length"],
        cache_dir=config["cache_dir"]
    )
    
    # 计算采样率
    dataset_size = len(imdb_datasets["train"])
    sample_rate = config["batch_size"] / dataset_size
    logger.info(f"Dataset size: {dataset_size}, Sample rate: {sample_rate:.6f}")
    
    # 创建训练数据加载器
    train_loader = DPDataLoader(
        dataset=imdb_datasets["train"],
        sample_rate=sample_rate,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(42) 
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        imdb_datasets["test"],  # IMDB使用test集作为验证
        batch_size=config["batch_size"],
        collate_fn=collator,
        shuffle=False
    )
    
    # 计算总训练步数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config["num_epochs"]
    logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    # 计算噪声乘数
    logger.info("Calculating noise multiplier...")
    noise_multiplier = get_noise_multiplier(
        target_epsilon=config["target_epsilon"],
        target_delta=config["target_delta"],
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp",
    )
    logger.info(f"Calculated noise multiplier: {noise_multiplier:.4f}")
    
    # 2. 加载模型   
    logger.info("Loading RoBERTa model for sentiment analysis...")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,  # 情感分析是二分类任务
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
    logger.info("Initializing privacy engine...")
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
    logger.info("Starting DP training for IMDB...")
    history, final_acc, final_epsilon = train_with_dpsgd(
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
    
    # 7. 最终评估和报告
    logger.info("\nTraining completed!")
    logger.info(f"Final validation accuracy: {final_acc:.4f}")
    logger.info(f"Final privacy consumption: ε={final_epsilon:.2f} (δ={config['target_delta']})")
    
    # 保存训练历史
    import json
    with open(os.path.join(config["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("All results saved to %s", config["output_dir"])

if __name__ == "__main__":
    main()