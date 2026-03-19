"""
Llama 预训练脚本。

用法:
  正常训练 (需要 GPU + 网络):
    python pretrain_llama.py

  快速验证模式 (仅 CPU，无需网络，几秒完成):
    python pretrain_llama.py --verify
"""

import argparse
import os
import sys
import shutil
import tempfile
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

parser = argparse.ArgumentParser(description="Llama pretraining script")
parser.add_argument("--verify", action="store_true",
                    help="Run in verification mode: tiny model, local data, CPU only, 5 steps")
args = parser.parse_args()

if args.verify:
    # ── 验证模式：离线、CPU、小模型、mock 数据 ──────────────────────
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    from datasets import Dataset

    print("=== 验证模式：使用本地 tokenizer 和 mock 数据 ===\n")

    # 1. 创建本地 BPE tokenizer（替代 gpt2，无需网络）
    corpus = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "Machine learning is a subset of artificial intelligence. " * 50,
        "Python is a great programming language for data science. " * 50,
        "Pre-training large language models requires significant compute resources. " * 50,
        "Natural language processing enables computers to understand human language. " * 50,
    ]
    base_tokenizer = Tokenizer(models.BPE())
    base_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    base_tokenizer.decoder = decoders.ByteLevel()
    tok_trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<pad>"],
        min_frequency=2,
    )
    base_tokenizer.train_from_iterator(corpus, trainer=tok_trainer)

    tokenizer_dir = tempfile.mkdtemp()
    base_tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"),
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    shutil.rmtree(tokenizer_dir, ignore_errors=True)
    print(f"本地 tokenizer 创建完成，vocab size: {len(tokenizer)}")

    # 2. 缩小版模型（2层，隐藏层64）
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    print(f"模型初始化完成。总参数量: {model.num_parameters() / 1e6:.2f} M")

    # 3. Mock 数据集
    mock_texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Machine learning is a subset of artificial intelligence. " * 5,
        "Python is a great programming language for data science. " * 5,
        "Pre-training large language models requires significant compute. " * 5,
    ]
    dataset = Dataset.from_dict({"text": mock_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. 训练参数（5步，CPU）
    training_args = TrainingArguments(
        output_dir="./verify-checkpoints",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n开始验证训练（5步）...")
    train_result = trainer.train()

    log_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    print(f"训练 loss 记录: {[f'{l:.4f}' for l in log_losses]}")

    # 5. 保存 & 重新加载 & 推理
    save_dir = "./verify-model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    loaded_model = LlamaForCausalLM.from_pretrained(save_dir)
    loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)

    input_text = "The future of"
    inputs = loaded_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        gen_output = loaded_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated_text = loaded_tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(f"推理验证 - 输入: {input_text!r} -> 输出: {generated_text!r}")

    # 清理
    shutil.rmtree("./verify-checkpoints", ignore_errors=True)
    shutil.rmtree(save_dir, ignore_errors=True)

    print("\n=== 验证通过！完整流水线正常：初始化 -> 数据处理 -> 训练 -> 保存/加载 -> 推理 ===")
    sys.exit(0)

else:
    # ── 正常预训练模式 ─────────────────────────────────────────────────
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # 1. 加载现成的开源 Tokenizer
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 从零初始化一个约 100M 参数的微型 Llama 架构
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    print(f"模型初始化完成。总参数量: {model.num_parameters() / 1e6:.2f} M")

    # 3. 流式加载海量高质量预训练数据
    print("正在连接数据集流...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    # 4. 数据预处理管道
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "id", "dump", "url", "file_path", "language", "language_score", "token_count"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./my-100m-llama-checkpoints",
        max_steps=50000,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        bf16=True,
        logging_steps=100,
        save_steps=2000,
        save_total_limit=3,
        dataloader_num_workers=2,
        report_to="none"
    )

    # 6. 启动 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("开始进行极其漫长但充满成就感的预训练...")
    trainer.train()

    # 7. 训练结束，保存最终模型
    trainer.save_model("./my-100m-llama-final")
    tokenizer.save_pretrained("./my-100m-llama-final")
    print("预训练完成！你的专属 100M 基座模型已保存。")
