import os
import logging
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # 通用配置
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"
    log_dir: str = "./logs"
    
    # 缓存配置
    cache_dir: str = "./cache"  # 缓存目录
    use_cache: bool = True      # 是否使用缓存
    cache_datasets: bool = True # 是否缓存处理后的数据集
    
    # 数据集配置
    dataset_name: str = "imdb"  # 可选: mnli, qnli, sst2, qqp, imdb
    data_dir: str = "./data"
    max_seq_length: int =  128
    batch_size: int = 32
    
    # 模型配置
    model_name: str = "roberta-base"
    num_labels: int = None  # 将由任务类型自动设置
    
    # 全参数微调配置
    full_param_lr: float = 1e-5
    full_param_epochs: int = 3    # 可自定义训练轮次
    save_per_epoch: bool = False  # 每轮保存断点
   
    # 子空间配置
    use_subspace: bool = True
    subspace_dim: int = 32                # 子空间维度，实际运行过程中会自动根据总步数和取样间隔计算出来
    use_full_svd: bool = False
    ensemble_size: int = 1              # 子空间集成大小 (h值)
    subspace_lr: float = 1e-2
    save_member_params: bool = False
    subspace_epochs: int = 8           # 子空间训练轮次
    warmup_ratio = 0.1                  # 预热阶段占总训练步数的比例
    scheduler_type = "linear"           # 调度器类型
    
    # 日志配置
    log_interval: int = 100     # 日志打印间隔（步数）
    progress_bar: bool = True   # 是否显示进度条
    progress_refresh: int = 4   # 进度条刷新间隔（秒）
    log_level: str = "INFO"     # 新增日志级别配置

    # 调试配置
    debug_mode: bool = True
    resume: bool = False

    # 新增跳过第一阶段配置
    skip_stage1: bool = False  # 是否跳过第一阶段训练
    stage1_snapshot_dir: str = None  # 第一阶段快照目录
    
    # 标签列名映射
    label_columns: dict = field(default_factory=lambda: {
        "mnli": "label",
        "qnli": "label",
        "sst2": "label",
        "qqp": "label",
        "mrpc": "label",
        "rte": "label",
        "wnli": "label",
        "cola": "label",
        "stsb": "label",
        "imdb": "label",
        "squad": "answers"  # SQuAD使用不同的标签结构
    })

    def setup_logger(self):
        logger = logging.getLogger("SubspaceTuning")
        logger.setLevel(logging.INFO)
        # 创建文件处理器
        file_handler = logging.FileHandler("subspace_tuning.log")
        file_handler.setLevel(logging.INFO)
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 将处理器添加到日志记录器
        logger.addHandler(file_handler)
        # 移除默认的控制台处理器
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
        return logger
    
    def __post_init__(self):
        # 根据数据集自动设置标签数量
        self.num_labels = self._get_num_labels_for_task()
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 设置stage1_snapshot_dir
        if self.stage1_snapshot_dir is None:
            # 自动生成合理的默认值
            self.stage1_snapshot_dir = os.path.join(
                self.output_dir,
                f"{self.model_name.replace('/', '-')}_{self.dataset_name}_{self.subspace_dim}"
            )
        
        # 配置日志系统
        self._setup_logging()
        
        # 打印配置信息
        self.logger.info("\n=== 实验配置 ===")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                self.logger.info(f"{key}: {value}")
        self.logger.info("===============\n")
    
    def _get_num_labels_for_task(self):
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
            "imdb": 2,       # 情感分析二分类
        }
        
        if self.dataset_name in task_label_map:
            return task_label_map[self.dataset_name]
        else:
            return 2
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志记录器
        self.logger = logging.getLogger('SubspaceTuning')
        log_level = getattr(logging, self.log_level.upper())
        self.logger.setLevel(log_level)
        
        # 创建文件处理器
        log_file = os.path.join(self.log_dir, f"{self.dataset_name}_training.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 移除所有处理器
        self.logger.handlers = []
        
        # 添加文件处理器到日志记录器
        self.logger.addHandler(file_handler)