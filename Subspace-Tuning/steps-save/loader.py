from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pickle
import hashlib
from functools import lru_cache

def get_cache_path(config, prefix=""):
    """生成缓存路径"""
    # 创建基于配置的唯一标识符
    config_hash = hashlib.md5(
        f"{config.dataset_name}_{config.model_name}_{config.max_seq_length}".encode()
    ).hexdigest()[:8]
    
    cache_path = os.path.join(
        config.cache_dir,
        f"{prefix}{config.dataset_name}_{config_hash}"
    )
    
    return cache_path

def load_model_and_tokenizer(config):
    """加载模型和 tokenizer，使用 transformers 内置缓存"""
    print(f"[LOADER] 加载模型: {config.model_name} (使用 transformers 内置缓存)")
    
    # 加载 tokenizer - 使用内置缓存
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )
    
    # 根据任务类型加载模型
    if config.dataset_name in ["mnli", "qnli", "sst2", "qqp", "mrpc", "rte", "wnli", "cola", "imdb"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            cache_dir=config.cache_dir
        )
    elif config.dataset_name == "squad":
        from transformers import AutoModelForQuestionAnswering
        model = AutoModelForQuestionAnswering.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
    elif config.dataset_name == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            problem_type="regression",
            cache_dir=config.cache_dir
        )
    else:
        raise ValueError(f"不支持的数据集: {config.dataset_name}")
    
    print(f"[LOADER] 模型加载完成 (使用缓存: {config.cache_dir})")
    return model, tokenizer

def load_raw_dataset(config):
    """
    使用 Hugging Face datasets 库内置的缓存机制加载数据集
    无需手动管理缓存，库会自动处理
    """
    print(f"[LOADER] 加载数据集: {config.dataset_name} (使用 datasets 内置缓存)")
    
    # 设置缓存目录
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # 根据数据集名称选择加载方式
    try:
        if config.dataset_name in ["mnli", "qnli", "sst2", "mrpc", "rte", "wnli", "cola"]:
            # GLUE 数据集
            dataset = load_dataset(
                "glue", 
                config.dataset_name,
                cache_dir=config.cache_dir  # 关键：使用内置缓存
            )
        elif config.dataset_name == "qqp":
            # QQP 数据集
            dataset = load_dataset(
                "glue", 
                "qqp",
                cache_dir=config.cache_dir
            )
        elif config.dataset_name == "squad":
            # SQuAD 数据集
            dataset = load_dataset(
                "squad",
                cache_dir=config.cache_dir
            )
        elif config.dataset_name == "stsb":
            # STS-B 数据集
            dataset = load_dataset(
                "glue", 
                "stsb",
                cache_dir=config.cache_dir
            )
        elif config.dataset_name == "imdb":
            # IMDB 数据集
            dataset = load_dataset(
                "imdb",
                cache_dir = config.cache_dir
            )
        else:
            raise ValueError(f"不支持的数据集: {config.dataset_name}")
    except Exception as e:
        print(f"[LOADER] 数据集加载失败: {str(e)}")
        raise
    
    print(f"[LOADER] 数据集加载完成 (使用缓存: {config.cache_dir})")
    return dataset