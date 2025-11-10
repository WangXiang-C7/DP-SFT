import os
import logging
import argparse
import torch
import time

def parse_arguments():
    """创建并解析命令行参数"""
    parser = argparse.ArgumentParser(description="子空间微调框架配置")
    
    # ========== 通用配置 ==========
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备 (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    # ========== 缓存配置 ==========
    parser.add_argument("--cache_dir", type=str, default="./cache", help="缓存目录")
    parser.add_argument("--use_cache", action="store_true", default=True, help="是否使用缓存")
    parser.add_argument("--cache_datasets", action="store_true", default=True, help="是否缓存处理后的数据集")
    parser.add_argument("--save_svd_plots", action="store_true", default=True, help="保存SVD分解的可视化结果")
    # ========== 数据集配置 ==========
    parser.add_argument("--dataset_name", type=str, default="imdb", choices=["mnli", "qnli", "sst2", "qqp", "mrpc", "rte", "wnli", "cola", "stsb", "squad","imdb"], help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集目录")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    # ========== 模型配置 ==========
    parser.add_argument("--model_name", type=str, default="roberta-base", help="预训练模型名称")
    # ========== 第一阶段：全参数微调配置 ==========
    parser.add_argument("--stage1_lr", type=float, default=5e-4, help="一阶段学习率")
    parser.add_argument("--stage1_epochs", type=int, default=2, help="一阶段训练轮次")
    parser.add_argument("--stage1_warmup_ratio", type=float, default=0.1, help="一阶段预热阶段占总训练步数的比例")
    parser.add_argument("--stage1_weight_decay", type=float, default=0.01, help="权重衰减")
    # 第一阶段DP配置
    parser.add_argument("--stage1_clip_norm", type=float, default=10.0, help="一阶段梯度裁剪范数")
    parser.add_argument("--stage1_target_epsilon", type=float, default=3.0, help="一阶段目标隐私预算ε")
    parser.add_argument("--stage1_target_delta", type=float, default=1e-5, help="一阶段目标松驰项δ")
    parser.add_argument("--stage1_noise_multiplier", type=float, default=None, help="一阶段噪声乘子σ")
    # ========== 第二阶段：子空间训练配置 ==========
    parser.add_argument("--stage2_subspace_dim", type=int, default=64, help="二阶段子空间维度")
    parser.add_argument("--stage2_lr", type=float, default=1e-5, help="二阶段学习率")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="二阶段训练轮次")
    parser.add_argument("--stage2_warmup_ratio", type=float, default=0.1, help="二阶段预热阶段占总训练步数的比例")
    parser.add_argument("--stage2_weight_decay", type=float, default=0.01, help="权重衰减")
    # 第二阶段DP配置         
    parser.add_argument("--stage2_clip_norm", type=float, default=0.5, help="二阶段梯度裁剪范数")
    parser.add_argument("--stage2_target_epsilon", type=float, default=4, help="二阶段目标隐私预算ε")
    parser.add_argument("--stage2_target_delta", type=float, default=1e-5, help="二阶段目标松弛项δ")
    parser.add_argument("--stage2_noise_multiplier", type=float, default=None, help="二阶段噪声乘子σ")
    # ========== 日志配置 ==========
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（步数）")
    parser.add_argument("--progress_bar", action="store_true", default=True, help="是否显示进度条")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")          
    # ========== 解析参数 ==========
    args = parser.parse_args()
    
    # 后处理：设置标签数量
    args.num_labels = get_num_labels_for_task(args.dataset_name)
    
    # 确保目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 配置日志系统
    args.logger = setup_logger(args)
    
    # 打印配置信息
    args.logger.info("\n=== 实验配置 ===")
    for arg in vars(args):
        if arg not in ['logger']:  # 排除logger本身
            args.logger.info(f"{arg}: {getattr(args, arg)}")
    args.logger.info("===============\n")
    
    return args

def get_num_labels_for_task(dataset_name):
    """根据任务类型自动确定标签数量"""
    task_label_map = {
        "mnli": 3,       # 自然语言推理：蕴含、中性、矛盾
        "qnli": 2,       # 问题自然语言推理：蕴含、不蕴含
        "sst2": 2,       # 情感分析：正面、负面
        "qqp": 2,        # 问题对相似度：相似、不相似
        "squad": 2,      # SQuAD问答：开始位置和结束位置
        "mrpc": 2,       # 文本相似度：相似、不相似
        "rte": 2,        # 文本蕴含：蕴含、不蕴含
        "wnli": 2,       # 词义消歧：蕴含、不蕴含
        "cola": 2,       # 语法正确性：可接受、不可接受
        "stsb": 1,       # 语义文本相似度：回归任务
        "imdb": 2,       # 情感分析：正面、负面
    }
    return task_label_map.get(dataset_name, 2)

def setup_logger(args):
    """配置日志系统，确保不重复添加处理器"""
    # 创建动态命名的日志文件
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"{args.model_name}_{args.dataset_name}_{timestamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)
    
    logger = logging.getLogger("SubspaceTuning")
    
    # 如果日志器已经配置过处理器，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 防止日志向上传播
    logger.propagate = False
    
    return logger