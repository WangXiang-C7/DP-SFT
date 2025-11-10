from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def prepare_datasets(max_length, cache_dir):
    dataset = load_dataset("glue", "mnli", cache_dir=cache_dir)
    
    # 2. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", cache_dir=cache_dir)
    
    def preprocess_function(examples):
        # 单句分词
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
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