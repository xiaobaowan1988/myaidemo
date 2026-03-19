"""
Llama 预训练脚本。

用法:
  正常训练 (需要 GPU + 网络):
    python pretrain_llama.py

  快速验证模式 (仅 CPU，无需网络，几秒完成):
    python pretrain_llama.py --verify
"""

import argparse
import shutil
import sys
import torch
from transformers import LlamaForCausalLM, Trainer

from pretrain_common import (
    create_model,
    create_data_collator,
    tokenize_dataset,
)

parser = argparse.ArgumentParser(description="Llama pretraining script")
parser.add_argument("--verify", action="store_true",
                    help="Run in verification mode: tiny model, local data, CPU only, 5 steps")
args = parser.parse_args()

if args.verify:
    # ── 验证模式：离线、CPU、小模型、mock 数据 ──────────────────────
    from pretrain_common import (
        create_local_tokenizer,
        create_mock_dataset,
        create_verify_training_args,
    )
    from transformers import PreTrainedTokenizerFast

    print("=== 验证模式：使用本地 tokenizer 和 mock 数据 ===\n")

    tokenizer = create_local_tokenizer()
    print(f"本地 tokenizer 创建完成，vocab size: {len(tokenizer)}")

    model = create_model(tokenizer)  # 默认即为 tiny 模型
    print(f"模型初始化完成。总参数量: {model.num_parameters() / 1e6:.2f} M")

    tokenized_dataset = create_mock_dataset(tokenizer)
    training_args = create_verify_training_args()
else:
    # ── 正常预训练模式 ─────────────────────────────────────────────────
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = create_model(tokenizer, hidden_size=768, intermediate_size=3072,
                         num_layers=12, num_heads=12, max_seq_len=1024)
    print(f"模型初始化完成。总参数量: {model.num_parameters() / 1e6:.2f} M")

    print("正在连接数据集流...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True)
    tokenized_dataset = tokenize_dataset(
        dataset, tokenizer, max_length=1024,
        remove_columns=["text", "id", "dump", "url", "file_path",
                         "language", "language_score", "token_count"],
    )

    from transformers import TrainingArguments
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
        report_to="none",
    )

# ── 以下为两个模式共用的训练 + 保存逻辑 ───────────────────────────
data_collator = create_data_collator(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

if args.verify:
    print("\n开始验证训练（5步）...")
else:
    print("开始进行极其漫长但充满成就感的预训练...")

trainer.train()

if args.verify:
    log_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    print(f"训练 loss 记录: {[f'{l:.4f}' for l in log_losses]}")

    # 保存 & 重新加载 & 推理验证
    save_dir = "./verify-model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    loaded_model = LlamaForCausalLM.from_pretrained(save_dir)
    from transformers import PreTrainedTokenizerFast
    loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)

    input_text = "The future of"
    inputs = loaded_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        gen_output = loaded_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated_text = loaded_tokenizer.decode(gen_output[0], skip_special_tokens=True)
    print(f"推理验证 - 输入: {input_text!r} -> 输出: {generated_text!r}")

    shutil.rmtree("./verify-checkpoints", ignore_errors=True)
    shutil.rmtree(save_dir, ignore_errors=True)
    print("\n=== 验证通过！完整流水线正常：初始化 -> 数据处理 -> 训练 -> 保存/加载 -> 推理 ===")
else:
    trainer.save_model("./my-100m-llama-final")
    tokenizer.save_pretrained("./my-100m-llama-final")
    print("预训练完成！你的专属 100M 基座模型已保存。")
