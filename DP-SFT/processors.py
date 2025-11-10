
import torch
import os
import pickle
import hashlib
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class DataProcessor(ABC):
    """数据集处理器基类"""
    @abstractmethod
    def process_dataset(self, dataset, tokenizer, config):
        """处理数据集"""
        pass


def get_processed_cache_path(config, processor_name):
    """生成处理后数据集的缓存路径"""
    # 创建基于配置的唯一标识符
    config_hash = hashlib.md5(
        f"{config.dataset_name}_{config.model_name}_{config.max_length}_{processor_name}".encode()
    ).hexdigest()[:8]

    cache_path = os.path.join(
        config.cache_dir,
        f"processed_{config.dataset_name}_{config_hash}"
    )

    return cache_path


class GLUEProcessor(DataProcessor):
    def _tokenize(self, examples, config, tokenizer):
        if config.dataset_name == "mnli":
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name == "qnli":
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name == "sst2":
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name == "qqp":
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name in ["mrpc", "rte", "wnli", "cola"]:
            return tokenizer(
                examples["sentence1"],
                examples.get("sentence2", None),
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name == "stsb":
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        elif config.dataset_name == "imdb":
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )

    def process_dataset(self, raw_dataset, tokenizer, config):
        """
        处理数据集，使用 datasets 库的 map 函数内置缓存
        """
        print(f"[PROCESSOR] 处理数据集: {config.dataset_name}")

        # 获取当前数据集对应的标签列名
        label_columns = {
            "mnli": "label",
            "qnli": "label",
            "sst2": "label",
            "qqp": "label",
            "mrpc": "label",
            "rte": "label",
            "wnli": "label",
            "cola": "label",
            "stsb": "label",
            "imdb": "label"
        }
        label_col = label_columns.get(config.dataset_name, "label")

        # 预处理函数
        def preprocess_function(examples):
            result = self._tokenize(examples, config, tokenizer)
            result["labels"] = examples[label_col]
            return result

        # 使用 datasets 的 map 函数处理，并启用内置缓存
        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=config.use_cache  # 启用内置缓存
        )

        # 设置格式为PyTorch张量
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        print(f"[PROCESSOR] 数据集处理完成 (使用缓存: {config.cache_dir})")
        return processed_dataset


class SquadProcessor(DataProcessor):
    """SQuAD数据集处理器"""
    def process_dataset(self, dataset, tokenizer, config):
        processor_name = "squad_processor"
        cache_path = get_processed_cache_path(config, processor_name)

        # 检查缓存
        if config.use_cache and config.cache_datasets and os.path.exists(cache_path):
            try:
                print(f"[PROCESSOR] 从缓存加载处理后的 SQuAD 数据集: {cache_path}")
                processed_dataset = pickle.load(open(cache_path, "rb"))
                return processed_dataset
            except Exception as e:
                print(f"[PROCESSOR] 从缓存加载数据失败: {e}")

        print(f"[PROCESSOR] 处理 SQuAD 数据集")

        def preprocess_function(examples):
            # 对问题和上下文进行编码
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                truncation="only_second",
                max_length=config.max_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # 处理答案
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # 找到序列的开始和结束
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # 如果答案不在上下文中，标记为(0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # 否则，找到答案的开始和结束位置
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = torch.tensor(start_positions)
            inputs["end_positions"] = torch.tensor(end_positions)
            return inputs

        # 应用预处理
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=[col for col in dataset["train"].column_names if col not in ["labels"]]
        )

        # 设置格式为PyTorch张量
        processed_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
        )

        # 保存到缓存
        if config.use_cache and config.cache_datasets:
            try:
                print(f"[PROCESSOR] 保存处理后的数据集到缓存: {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                pickle.dump(processed_dataset, open(cache_path, "wb"))
            except Exception as e:
                print(f"[PROCESSOR] 保存数据到缓存失败: {e}")

        print(f"[PROCESSOR] 数据集处理完成")
        return processed_dataset


def get_processor(dataset_name):
    """根据数据集名称获取对应的处理器"""
    processors = {
        "mnli": GLUEProcessor,
        "qnli": GLUEProcessor,
        "sst2": GLUEProcessor,
        "qqp": GLUEProcessor,
        "mrpc": GLUEProcessor,
        "rte": GLUEProcessor,
        "wnli": GLUEProcessor,
        "cola": GLUEProcessor,
        "stsb": GLUEProcessor,
        "squad": SquadProcessor,
        "imdb": GLUEProcessor,
    }
    return processors[dataset_name]()