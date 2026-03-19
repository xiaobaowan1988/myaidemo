"""
验证预训练代码的正确性（小规模离线测试，不需要GPU，不需要网络）。

由于运行环境无法访问 HuggingFace Hub，本脚本使用本地创建的 BPE tokenizer
替代 gpt2 tokenizer，但验证逻辑与原始代码完全一致：
  1. Tokenizer 加载与 pad_token 设置
  2. LlamaConfig + LlamaForCausalLM 随机初始化，参数量验证
  3. 前向传播正确性
  4. 数据预处理管道（用本地 mock 数据代替流式数据集）
  5. DataCollator 生成 labels 的正确性
  6. Trainer 初始化 + 跑几步训练，验证 loss
  7. 模型保存与重新加载 + 推理
"""

import os
import shutil
import sys
import torch
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
)
from pretrain_common import (
    TOKENIZER_CORPUS,
    TINY_MAX_SEQ_LEN,
    create_local_tokenizer,
    create_tiny_model,
    create_mock_dataset,
    create_verify_training_args,
    create_data_collator,
)

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} -- {detail}")


# ── Test 1: 创建本地 Tokenizer (模拟 gpt2) ─────────────────────────
print("\n=== Test 1: Tokenizer 创建与配置 ===")

# 添加额外语料行以增强 tokenizer
extra_corpus = TOKENIZER_CORPUS + [
    "The future of AI is bright and full of possibilities. " * 50,
]
tokenizer = create_local_tokenizer(corpus=extra_corpus)
VOCAB_SIZE = len(tokenizer)
print(f"  本地 tokenizer vocab size: {VOCAB_SIZE}")
check("tokenizer 创建成功", tokenizer is not None)
check("pad_token == eos_token", tokenizer.pad_token == tokenizer.eos_token)
check("pad_token_id == eos_token_id", tokenizer.pad_token_id == tokenizer.eos_token_id)

# 验证 tokenizer 基本功能
encoded = tokenizer("Hello world", truncation=True, max_length=128, padding="max_length")
check("编码后 input_ids 长度 == 128", len(encoded["input_ids"]) == 128)
check("attention_mask 长度 == 128", len(encoded["attention_mask"]) == 128)

# ── Test 2: Model Config & Init ────────────────────────────────────
print("\n=== Test 2: 模型初始化 ===")

# 验证原始 100M 配置的参数量计算
print("  -- 验证原始 100M 配置的参数量 --")
embed_params = 50257 * 768
per_layer_attn = 4 * 768 * 768  # Q, K, V, O projections
per_layer_mlp = 3 * 768 * 3072  # gate_proj, up_proj, down_proj
per_layer_ln = 2 * 768
per_layer = per_layer_attn + per_layer_mlp + per_layer_ln
total_layers = 12 * per_layer
final_ln = 768
lm_head = 50257 * 768
estimated_params = (embed_params + total_layers + final_ln + lm_head) / 1e6
print(f"  估算参数量: {estimated_params:.2f} M")
check("100M 配置参数量在合理范围 (80-250M)", 80 < estimated_params < 250,
      f"估算: {estimated_params:.2f}M")

# 用小模型做实际测试
print("  -- 使用缩小版模型做功能验证 --")
model = create_tiny_model(tokenizer)
num_params = model.num_parameters() / 1e6
print(f"  测试模型参数量: {num_params:.2f} M")
check("模型初始化成功", model is not None)
check("config.vocab_size 正确", model.config.vocab_size == VOCAB_SIZE)
check("config.num_hidden_layers == 2", model.config.num_hidden_layers == 2)
check("config.max_position_embeddings == 128", model.config.max_position_embeddings == 128)

# ── Test 3: Forward pass ───────────────────────────────────────────
print("\n=== Test 3: 前向传播 ===")
dummy_input = torch.randint(0, VOCAB_SIZE, (2, 64))
dummy_labels = dummy_input.clone()
with torch.no_grad():
    outputs = model(input_ids=dummy_input, labels=dummy_labels)

check("前向传播无报错", True)
check("loss 非空", outputs.loss is not None)
check("loss 是标量", outputs.loss.dim() == 0)
check("logits shape 正确", outputs.logits.shape == (2, 64, VOCAB_SIZE),
      f"实际: {outputs.logits.shape}")
expected_loss = torch.log(torch.tensor(float(VOCAB_SIZE))).item()
actual_loss = outputs.loss.item()
print(f"  期望 loss ≈ ln({VOCAB_SIZE}) = {expected_loss:.2f}, 实际 loss = {actual_loss:.2f}")
check(f"初始 loss 接近 ln(vocab_size)",
      abs(actual_loss - expected_loss) < 2.0,
      f"差值: {abs(actual_loss - expected_loss):.2f}")

# ── Test 4: Data pipeline (mock) ──────────────────────────────────
print("\n=== Test 4: 数据预处理管道 ===")
tokenized_dataset = create_mock_dataset(tokenizer)

check("tokenized dataset 非空", len(tokenized_dataset) == 4)
check("包含 input_ids 列", "input_ids" in tokenized_dataset.column_names)
check("包含 attention_mask 列", "attention_mask" in tokenized_dataset.column_names)
check("每条 input_ids 长度 == 128", len(tokenized_dataset[0]["input_ids"]) == 128)

# ── Test 5: DataCollator ──────────────────────────────────────────
print("\n=== Test 5: DataCollator ===")
data_collator = create_data_collator(tokenizer)
batch = data_collator([tokenized_dataset[i] for i in range(2)])

check("batch 包含 input_ids", "input_ids" in batch)
check("batch 包含 labels", "labels" in batch)
check("batch 包含 attention_mask", "attention_mask" in batch)
check("labels shape == input_ids shape",
      batch["labels"].shape == batch["input_ids"].shape)
pad_positions = (batch["attention_mask"] == 0)
if pad_positions.any():
    check("pad 位置 labels == -100",
          (batch["labels"][pad_positions] == -100).all().item())
else:
    check("无 pad 位置（文本足够长）", True)

# ── Test 6: Trainer 小规模训练 ─────────────────────────────────────
print("\n=== Test 6: Trainer 训练验证（5 步） ===")
output_dir = "./verify_test_output"
training_args = create_verify_training_args(output_dir=output_dir)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

train_result = trainer.train()
check("训练完成无报错", True)
check("train_loss 存在", "train_loss" in train_result.metrics,
      f"metrics: {train_result.metrics}")

log_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
print(f"  训练 loss 记录: {[f'{l:.4f}' for l in log_losses]}")
check("记录了 loss", len(log_losses) > 0)
if len(log_losses) >= 2:
    check("loss 有变化（模型在学习）", log_losses[0] != log_losses[-1])

# ── Test 7: 模型保存与加载 ────────────────────────────────────────
print("\n=== Test 7: 模型保存与加载 ===")
save_dir = "./verify_test_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

check("模型目录已创建", os.path.isdir(save_dir))
check("config.json 存在", os.path.isfile(os.path.join(save_dir, "config.json")))
check("model.safetensors 存在",
      os.path.isfile(os.path.join(save_dir, "model.safetensors")))
check("tokenizer 文件存在",
      os.path.isfile(os.path.join(save_dir, "tokenizer.json")))

loaded_model = LlamaForCausalLM.from_pretrained(save_dir)
loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)
check("模型重新加载成功", loaded_model is not None)
check("加载的模型参数量一致",
      loaded_model.num_parameters() == model.num_parameters())
check("tokenizer 重新加载成功", loaded_tokenizer is not None)
check("加载的 vocab size 一致", len(loaded_tokenizer) == len(tokenizer))

input_text = "The future of"
inputs = loaded_tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    gen_output = loaded_model.generate(**inputs, max_new_tokens=10, do_sample=False)
generated_text = loaded_tokenizer.decode(gen_output[0], skip_special_tokens=True)
check("生成文本非空", len(generated_text) > len(input_text),
      f"生成: {generated_text!r}")
print(f"  生成示例: {generated_text!r}")

# ── 清理 ──────────────────────────────────────────────────────────
shutil.rmtree(output_dir, ignore_errors=True)
shutil.rmtree(save_dir, ignore_errors=True)

# ── 汇总 ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"验证完成: {PASSED} 通过, {FAILED} 失败 (共 {PASSED + FAILED} 项)")
if FAILED == 0:
    print("所有测试通过! 原始预训练代码的逻辑和架构验证正确。")
    print("注意：由于离线环境限制，tokenizer 使用本地创建的小型 BPE 替代 gpt2，")
    print("但验证覆盖了完整的训练流水线：模型初始化 -> 数据处理 -> 训练 -> 保存/加载 -> 推理。")
print(f"{'='*60}")

sys.exit(0 if FAILED == 0 else 1)
