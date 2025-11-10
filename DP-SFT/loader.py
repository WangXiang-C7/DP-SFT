import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    default_data_collator
)
from torch.utils.data import DataLoader
from config import parse_arguments
import os
import hashlib
import config as con

class DPRandomBatchLoader:
    """支持DP-SGD的随机批次加载器（确定性无放回采样）"""
    def __init__(self, dataset, tokenizer, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_size = len(dataset)
        
        # 创建动态填充器（问答任务需要特殊collator）
        if config.dataset_name == "squad":
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
        # 创建随机状态生成器
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)
        self.current_step = 0
        # 预计算所有索引
        self.all_indices = torch.arange(self.dataset_size)
        
    def get_batch(self, step=None):
        """获取指定step的随机批次（无放回采样）"""
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        # 设置确定性随机种子（每次step增加种子）
        current_seed = self.config.seed + step
        generator = torch.Generator()
        generator.manual_seed(current_seed)
        
        # 随机排列索引（无放回）
        indices = torch.randperm(
            self.dataset_size, 
            generator=generator
        )[:self.config.batch_size].tolist()
        
        # 获取样本并组装批次
        samples = [self.dataset[i] for i in indices]
        batch = self.data_collator(samples)
            
        return {k: v.to(self.config.device) for k, v in batch.items()}

def get_cache_path(config, prefix=""):
    """生成缓存路径"""
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
    config.logger.info(f"[LOADER] 加载模型: {config.model_name}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )
    
    # 根据任务类型加载模型
    if config.dataset_name in ["mnli", "qnli", "sst2", "qqp", "mrpc", "rte", "wnli", "cola","imdb"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=con.get_num_labels_for_task(config.dataset_name),
            cache_dir=config.cache_dir
        )
    elif config.dataset_name == "squad":
        model = AutoModelForQuestionAnswering.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
    elif config.dataset_name == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=1,  # STS-B是回归任务
            cache_dir=config.cache_dir
        )
    else:
        raise ValueError(f"不支持的数据集: {config.dataset_name}")
    
    config.logger.info(f"[LOADER] 模型加载完成")
    return model, tokenizer

def load_raw_dataset(config):
    """使用 Hugging Face datasets 库加载数据集"""
    config.logger.info(f"[LOADER] 加载数据集: {config.dataset_name}")
    os.makedirs(config.cache_dir, exist_ok=True)
    
    try:
        if config.dataset_name in ["mnli", "qnli", "sst2", "mrpc", "rte", "wnli", "cola", "qqp", "stsb"]:
            dataset = load_dataset(
                "glue", 
                config.dataset_name,
                cache_dir=config.cache_dir
            )
        elif config.dataset_name == "imdb":
            dataset = load_dataset(
            "imdb",
            cache_dir=config.cache_dir
        )
        elif config.dataset_name == "squad":
            dataset = load_dataset(
                "squad",
                cache_dir=config.cache_dir
            )
        else:
            raise ValueError(f"不支持的数据集: {config.dataset_name}")
    except Exception as e:
        config.logger.error(f"数据集加载失败: {str(e)}")
        raise
    
    config.logger.info(f"[LOADER] 数据集加载完成")
    return dataset

def preprocess_dataset(dataset, tokenizer, config):
    """数据集预处理函数（支持分类和问答任务）"""
    def tokenize_function(examples):
        # 分类任务处理
        if config.dataset_name != "squad":
            # 根据数据集名称处理字段差异
            if config.dataset_name == "mnli":
                text1 = examples["premise"]
                text2 = examples["hypothesis"]
            elif config.dataset_name == "qnli":
                text1 = examples["question"]
                text2 = examples["sentence"]
            elif config.dataset_name == "sst2":
                text1 = examples["sentence"]
                text2 = None
            elif config.dataset_name == "qqp":
                text1 = examples["question1"]
                text2 = examples["question2"]
            elif config.dataset_name in ["mrpc", "rte", "wnli", "cola", "stsb"]:
                text1 = examples["sentence1"]
                text2 = examples.get("sentence2", None)
            elif config.dataset_name == "imdb":
                text1 = examples["text"]
                text2 = None
            else:
                raise ValueError(f"不支持的数据集: {config.dataset_name}")
            
            # 执行分词
            tokenized = tokenizer(
                text1,
                text2,
                truncation=True,
                max_length=config.max_length
            )
            
            # 添加标签字段（所有分类/回归任务）
            tokenized["labels"] = examples["label"]
            return tokenized

        # 问答任务处理
        else:
            tokenized = tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=config.max_seq_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            
            # 处理答案位置
            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized.pop("offset_mapping")
            
            tokenized["start_positions"] = []
            tokenized["end_positions"] = []
            
            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized["input_ids"][i]
                sequence_ids = tokenized.sequence_ids(i)
                
                # 找到上下文开始位置
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                
                # 找到上下文结束位置
                idx = len(input_ids) - 1
                while sequence_ids[idx] != 1:
                    idx -= 1
                context_end = idx
                
                # 获取原始样本
                sample_index = sample_mapping[i]
                answers = examples["answers"][sample_index]
                
                # 如果没有答案，使用CLS位置
                if len(answers["answer_start"]) == 0:
                    tokenized["start_positions"].append(0)
                    tokenized["end_positions"].append(0)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])
                    
                    # 查找token索引
                    token_start_index = context_start
                    while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    token_start_index -= 1
                    
                    token_end_index = context_end
                    while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    token_end_index += 1
                    
                    tokenized["start_positions"].append(token_start_index)
                    tokenized["end_positions"].append(token_end_index)
            
            return tokenized
    
    # 应用预处理
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_datasets

def create_data_loaders(config, tokenizer):
    """创建DP-SGD数据加载器"""
    # 1. 加载原始数据集
    raw_dataset = load_raw_dataset(config)
    
    # 2. 预处理数据集
    processed_dataset = preprocess_dataset(raw_dataset, tokenizer, config)
    
    # 3. 确定训练和评估数据集
    train_dataset = processed_dataset["train"]
    if config.dataset_name == "mnli":
        eval_dataset = processed_dataset["validation_matched"]
    elif config.dataset_name == "imdb":  
        eval_dataset = processed_dataset["test"]
    else:
        eval_dataset = processed_dataset["validation"]
    
    # 4. 创建评估数据加载器（使用与训练相同的collator逻辑）
    if config.dataset_name == "squad":
        eval_collator = default_data_collator
    else:
        eval_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=eval_collator
    )
    
    # 5. 创建DP训练加载器
    config.logger.info("[LOADER] 创建DP随机批次加载器")
    train_loader = DPRandomBatchLoader(train_dataset, tokenizer, config)

    # 6. 根据任务类型设置损失函数
    if config.dataset_name == "stsb":
        # 回归任务使用MSE损失
        loss_fn = torch.nn.MSELoss(reduction='none')
    else:
        # 分类任务使用交叉熵损失
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    return train_loader, eval_loader, loss_fn
