"""
预训练脚本公共模块：本地 tokenizer 创建、小模型配置、mock 数据集等。

被 pretrain_llama.py (--verify 模式) 和 verify_pretrain.py 共用。
"""

import os
import shutil
import tempfile

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# ── 训练语料（用于训练本地 BPE tokenizer） ──────────────────────────
TOKENIZER_CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 50,
    "Machine learning is a subset of artificial intelligence. " * 50,
    "Python is a great programming language for data science. " * 50,
    "Pre-training large language models requires significant compute resources. " * 50,
    "Natural language processing enables computers to understand human language. " * 50,
]

# ── Mock 文本（用于小规模训练验证） ─────────────────────────────────
MOCK_TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 5,
    "Machine learning is a subset of artificial intelligence. " * 5,
    "Python is a great programming language for data science. " * 5,
    "Pre-training large language models requires significant compute. " * 5,
]

# ── 小模型超参数 ───────────────────────────────────────────────────
TINY_HIDDEN_SIZE = 64
TINY_INTERMEDIATE_SIZE = 128
TINY_NUM_LAYERS = 2
TINY_NUM_HEADS = 4
TINY_MAX_SEQ_LEN = 128
TOKENIZER_VOCAB_SIZE = 512


def create_local_tokenizer(corpus=None):
    """创建本地 BPE tokenizer（离线，无需网络）。

    Returns:
        PreTrainedTokenizerFast 实例
    """
    if corpus is None:
        corpus = TOKENIZER_CORPUS

    base_tokenizer = Tokenizer(models.BPE())
    base_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base_tokenizer.decoder = decoders.ByteLevel()
    tok_trainer = trainers.BpeTrainer(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<pad>"],
        min_frequency=2,
    )
    base_tokenizer.train_from_iterator(corpus, trainer=tok_trainer)

    tokenizer_dir = tempfile.mkdtemp()
    try:
        tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
        base_tokenizer.save(tok_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            eos_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )
    finally:
        shutil.rmtree(tokenizer_dir, ignore_errors=True)

    return tokenizer


def create_tiny_model(tokenizer):
    """创建缩小版 Llama 模型（2层，hidden_size=64），用于快速验证。"""
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=TINY_INTERMEDIATE_SIZE,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=TINY_NUM_HEADS,
        num_key_value_heads=TINY_NUM_HEADS,
        max_position_embeddings=TINY_MAX_SEQ_LEN,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(config)


def create_mock_dataset(tokenizer, texts=None, max_length=TINY_MAX_SEQ_LEN):
    """创建 mock 数据集并 tokenize。"""
    if texts is None:
        texts = MOCK_TEXTS
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def create_verify_training_args(output_dir="./verify-checkpoints"):
    """创建验证模式的 TrainingArguments（5步，CPU）。"""
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_steps=5,
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=True,
    )


def create_data_collator(tokenizer):
    """创建 CLM DataCollator。"""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
