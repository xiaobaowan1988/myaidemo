import os
import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 1. 加载现成的开源 Tokenizer
# 使用 gpt2，无需申请权限，词汇表大小 50257，非常适合约 100M 参数的预算
tokenizer_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# GPT-2 没有 pad token，我们用 eos token 替代
tokenizer.pad_token = tokenizer.eos_token

# 2. 从零初始化一个约 100M 参数的微型 Llama 架构
# 经典配置：12 层，12 头，隐藏层 768
config = LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=12,
    max_position_embeddings=1024, # 上下文窗口长度设为 1024
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = LlamaForCausalLM(config) # 完全随机初始化的权重！
print(f"模型初始化完成。总参数量: {model.num_parameters() / 1e6:.2f} M")

# 3. 流式加载海量高质量预训练数据
# 使用 HuggingFaceFW/fineweb-edu，streaming=True 保证内存不爆
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

# 用 map 绑定数据流和分词器
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    # 丢弃非必须的列，提高数据流转效率
    remove_columns=["text", "id", "dump", "url", "file_path", "language", "language_score", "token_count"]
)

# DataCollator 会自动帮我们把 labels 设置为 input_ids，用于文字接龙预测
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. 配置训练参数 (重点)
training_args = TrainingArguments(
    output_dir="./my-100m-llama-checkpoints",
    max_steps=50000,                # 流式数据集没有明确的 epoch 概念，直接跑 50000 步
    per_device_train_batch_size=16, # 根据你的显存调整 (24GB 显存可设为 16 或 32)
    gradient_accumulation_steps=4,  # 累积梯度，用时间换空间，实际 Batch Size = 16 * 4 = 64
    learning_rate=5e-4,             # 预训练的 LR 通常比微调 (SFT) 稍微大一点
    weight_decay=0.01,
    bf16=True,                      # 开启 BF16 混合精度加速 (要求显卡是 3090/A10G/A100 等 Ampere 架构)
    logging_steps=100,              # 每 100 步在控制台打印一次 Loss (看着 Loss 下降是最爽的)
    save_steps=2000,                # 每 2000 步保存一次 Checkpoint，应对突发宕机
    save_total_limit=3,             # 最多保留最近的 3 个 Checkpoint 释放硬盘空间
    dataloader_num_workers=2,       # 开启多进程数据加载，防止 GPU 等待 CPU 数据喂入
    report_to="none"                # 纯净版控制台输出
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
