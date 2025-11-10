from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def prepare_datasets(max_length, cache_dir):
    # 1. 加载SST-2数据集
    dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
    
    # 2. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=cache_dir)
    
    # 3. 预处理函数（适配SST-2）
    def preprocess_function(examples):
        # 单句分词
        tokenized = tokenizer(
            examples["sentence"],  # SST-2的文本字段
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        # SST-2标签字段是"label"，值域0-1
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