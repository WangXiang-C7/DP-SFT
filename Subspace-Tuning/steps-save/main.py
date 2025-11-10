import os
# 在加载任何Hugging Face资源前设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from config import Config
from loader import load_model_and_tokenizer, load_raw_dataset
from processors import get_processor
from full_param_trainer import FullParamTrainer
from svd import perform_svd
from subspace_trainer import StrictSubspaceTrainer

def main():
    # 初始化配置 - 直接在代码中设置，不使用命令行参数
    config = Config()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    config.logger.info(f"\n===== 开始 {config.dataset_name} 实验 =====")
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
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
        # IMDB使用test集作为验证集
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["test"]
    else:
        train_dataset = processed_dataset["train"]
        eval_dataset = processed_dataset["validation"]
    
    config.logger.info(f"\n[MAIN] 数据集准备完成:")
    config.logger.info(f"  训练集: {len(train_dataset)} 样本")
    config.logger.info(f"  评估集: {len(eval_dataset)} 样本")

    # 如果配置跳过第一阶段
    if config.skip_stage1:
        config.logger.info("\n[MAIN] 跳过第一阶段训练...")
        snapshot_dir = config.stage1_snapshot_dir
        
        # 确保目录存在
        if not os.path.exists(snapshot_dir):
            config.logger.error(f"[MAIN] 错误: 快照目录不存在: {snapshot_dir}")
            raise FileNotFoundError(f"快照目录不存在: {snapshot_dir}")
        
        # 直接使用stage1_snapshot_dir作为轨迹目录
        trajectory_dir = snapshot_dir
        
        # 加载参数形状
        param_shapes_path = os.path.join(trajectory_dir, "param_shapes.pt")
        if not os.path.exists(param_shapes_path):
            config.logger.error(f"[MAIN] 错误: 参数形状文件不存在: {param_shapes_path}")
            raise FileNotFoundError(f"参数形状文件不存在: {param_shapes_path}")
        
        param_shapes = torch.load(param_shapes_path)
        
        # 获取所有轨迹文件并排序
        trajectory_files = sorted(
            [f for f in os.listdir(trajectory_dir) if f.endswith('.pt') and f != "param_shapes.pt" and f != "index.pt"],
            key=lambda x: int(x.split('.')[0])
        )
        
        # 加载所有轨迹点
        trajectory_list = []
        for file in trajectory_files:
            file_path = os.path.join(trajectory_dir, file)
            data = torch.load(file_path)
            
            # 展平所有参数
            flat_params = torch.cat([param.flatten() for param in data["params"].values()])
            trajectory_list.append(flat_params)
        
        # 堆叠成轨迹矩阵
        trajectory_matrix = torch.stack(trajectory_list)
        config.logger.info(f"[MAIN] 从 {trajectory_dir} 加载轨迹矩阵，形状: {trajectory_matrix.shape}")
        
        # 进行SVD分解
        principal_directions = perform_svd(trajectory_matrix, config)
        
        # 设置初始参数
        initial_params = trajectory_list[0]
        
        config.logger.info(f"[MAIN] 从 {snapshot_dir} 加载元数据")
    else:
        # 正常执行第一阶段训练
        config.logger.info("\n[MAIN] 开始第一阶段训练...")
        # 计算轨迹保存间隔
        total_steps = len(train_dataset) // config.batch_size * config.full_param_epochs
        trajectory_save_interval = max(1, total_steps // config.subspace_dim)
        # 第一阶段训练
        full_param_trainer = FullParamTrainer(config, train_dataset, eval_dataset)
        model, trajectory_matrix, initial_params, param_shapes = full_param_trainer.train_with_trajectory(
            model, 
            trajectory_save_interval=trajectory_save_interval
        )
        # SVD阶段
        principal_directions = perform_svd(trajectory_matrix, config)
    
    # 子空间训练
    strict_subspace_trainer = StrictSubspaceTrainer(config, train_dataset, eval_dataset)
    final_metric = strict_subspace_trainer.train(
        model, principal_directions, initial_params, param_shapes
    )
    
    config.logger.info(f"\n===== {config.dataset_name} 实验完成 =====")
    config.logger.info(f"[MAIN] 最终评估指标: {final_metric:.4f}")

if __name__ == "__main__":
    main()