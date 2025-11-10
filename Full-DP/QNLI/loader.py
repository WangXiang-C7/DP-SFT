from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def prepare_datasets(max_length, cache_dir):
    # 1. 加载QNLI数据集
    dataset = load_dataset("glue", "qnli", cache_dir=cache_dir)
    
    # 2. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=cache_dir)
    
    # 3. 预处理函数（适配QNLI）
    def preprocess_function(examples):
        # 拼接问题和句子作为输入
        texts = [f"Question: {q} Sentence: {s}" for q, s in zip(examples["question"], examples["sentence"])]
        
        tokenized = tokenizer(
            texts,  # QNLI的输入格式
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        # QNLI标签字段是"label"，值域0-1
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