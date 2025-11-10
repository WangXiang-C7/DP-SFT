import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

class DPSGDDataLoader:
    """DPSGD数据加载器，按step生成随机批次"""
    def __init__(self, max_length=128, batch_size=32, seed=42, cache_dir=".cache"):
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed + 1
        self.cache_dir = cache_dir
        
        # 准备数据集
        self.dataset, self.tokenizer, self.data_collator = self._prepare_datasets()
        self.dataset_size = len(self.dataset["train"])
    
    def _prepare_datasets(self):
        """准备数据集"""
        # 1. 加载数据集
        dataset = load_dataset("imdb", cache_dir=self.cache_dir)
        
        # 2. 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=self.cache_dir)
        
        # 3. 预处理函数 - 包含标签
        def preprocess_function(examples):
            # 分词处理
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            # 添加标签
            tokenized["labels"] = examples["label"]
            return tokenized
        
        # 4. 应用预处理
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # 5. 创建动态填充器
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        return tokenized_datasets, tokenizer, data_collator
    
    def get_batch(self, step):
        """获取指定step的随机批次（无放回采样）"""
        # 创建生成器并设置种子
        generator = torch.Generator()
        generator.manual_seed(self.seed + step)
        
        # 随机采样索引
        indices = torch.randperm(self.dataset_size, generator=generator)[:self.batch_size].tolist()
        
        # 获取样本并组装批次
        samples = [self.dataset["train"][i] for i in indices]
        batch = self.data_collator(samples)
        
        return batch

# # 临时检查是否是随机采样
# if __name__ == "__main__":
#     # 初始化DPSGD数据加载器
#     loader = DPSGDDataLoader(
#         max_length=128,
#         batch_size=64,
#         seed=42
#     )
    
#     batch = loader.get_batch(1)
#     print("Batch keys:", batch.keys())
#     print("Input IDs shape:", batch["input_ids"].shape)
#     print("Labels shape:", batch["labels"].shape)
#     print(batch)