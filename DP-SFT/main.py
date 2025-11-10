import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from loader import load_model_and_tokenizer, load_raw_dataset, create_data_loaders
from opacus import GradSampleModule
from processors import get_processor
from full_param_trainer import FullParamTrainer
from svd import perform_svd
from subspace_trainer import SubspaceDPTrainer
from config import parse_arguments
from dpsgd import calculate_sigma

def main():
    # 初始化配置
    config = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    config.logger.info(f"\n===== 开始 {config.dataset_name} 实验 =====")

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    model = model.to(config.device)
    model.train()
    # 加载原始数据集
    raw_dataset = load_raw_dataset(config)
    
    # 处理数据集
    processor = get_processor(config.dataset_name)
    processed_dataset = processor.process_dataset(raw_dataset, tokenizer, config)
    
    # 确定训练和评估数据集
    if config.dataset_name == "mnli":
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["validation_matched"]
    elif config.dataset_name == "imdb":
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["test"]
    else:
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["validation"]
    
    config.logger.info(f"\n[MAIN] 数据集准备完成:")
    config.logger.info(f"  训练集: {len(train_dataset)} 样本")
    config.logger.info(f"  评估集: {len(eval_dataset)} 样本")
    
    # 使用 loader.py 中的接口创建数据加载器
    train_loader, eval_loader, loss_fn = create_data_loaders(config = config, tokenizer = tokenizer)
    # ===============================================
    # 动态计算关键参数
    # ===============================================
    # 1. 计算第一阶段总步数
    stage1_total_steps = config.stage1_epochs * (len(train_dataset) // config.batch_size)
    config.logger.info(f"[MAIN] 第一阶段总步数: {stage1_total_steps}")
    
    # 2. 计算轨迹保存间隔
    trajectory_save_interval = stage1_total_steps // config.stage2_subspace_dim
    config.logger.info(f"[MAIN] 轨迹保存间隔: {trajectory_save_interval} 步")
    
    # 3. 计算一阶段噪声乘数
    sample_rate = config.batch_size / len(train_dataset)
    config.stage1_noise_multiplier = calculate_sigma(
        target_epsilon=config.stage1_target_epsilon,
        target_delta=config.stage1_target_delta,
        sample_rate=sample_rate,
        total_steps=stage1_total_steps
    )
    config.logger.info(f"[MAIN] 第一阶段噪声乘数: {config.stage1_noise_multiplier:.4f}")
    
    # 4. 计算第二阶段总步数及噪声乘数
    stage2_total_steps = config.stage2_epochs * (len(train_dataset) // config.batch_size)
    config.logger.info(f"[MAIN] 第二阶段总步数: {stage2_total_steps}")
    config.stage2_noise_multiplier = calculate_sigma(
        target_epsilon=config.stage2_target_epsilon,
        target_delta=config.stage2_target_delta,
        sample_rate=sample_rate,
        total_steps=stage2_total_steps
    )
    config.logger.info(f"[MAIN] 第二阶段噪声乘数: {config.stage2_noise_multiplier:.4f}")
    # ===============================================
    # 阶段1: 全参数微调并保存轨迹
    # ===============================================
    # config.logger.info(f"\n[MAIN] 开始全参数微调阶段 (轨迹保存间隔: {trajectory_save_interval} steps)...")
    # full_param_trainer = FullParamTrainer(config, train_loader, eval_loader, loss_fn)
    # model, trajectory_dir, param_shapes = full_param_trainer.train_with_trajectory(model)
    # ===============================================
    # 阶段2: SVD分解
    # ===============================================
    config.logger.info("\n[MAIN] 开始SVD分解阶段...")
    # 加载一个模型实例用于参数顺序
    trajectory_dir = "results/trajectory/roberta-base_imdb_20250710_110016"
    principal_directions = perform_svd(trajectory_dir, config, model)
    param_shapes = torch.load(os.path.join(trajectory_dir, "param_shapes.pt"))
    # ===============================================
    # 阶段3: 子空间微调
    # ===============================================
    config.logger.info("\n[MAIN] 开始子空间训练阶段...")
    subspace_dp_trainer = SubspaceDPTrainer(config, train_loader, eval_loader, loss_fn)
    model, final_metric = subspace_dp_trainer.train(
        model, principal_directions, param_shapes
    )
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), f"model/{config.dataset_name}-{config.stage2_subspace_dim}.pt")
    config.logger.info(f"\n===== {config.dataset_name} 实验完成 =====")
    config.logger.info(f"[MAIN] 最终评估指标: {final_metric:.4f}")

if __name__ == "__main__":
    main()