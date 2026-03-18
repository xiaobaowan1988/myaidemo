#!/usr/bin/env python3
"""
玩具级大模型训练全流程 Demo (Toy-level LLM Training Pipeline)
=============================================================

在普通笔记本 CPU 上几分钟内跑通大模型训练的四个核心阶段：
  1. 数据准备 (Data Preparation) — 清洗 + Tokenization
  2. 预训练 (Pre-training) — Next-Token Prediction
  3. 指令微调 (Supervised Fine-Tuning, SFT)
  4. 偏好对齐 (Alignment via DPO)

模型规格：~5M 参数的微型 GPT (2 层 Transformer Decoder)
运行环境：纯 PyTorch，CPU 即可，无需 GPU
"""

import math
import random
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 全局配置
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 256        # 字节级词表 (Byte-level)
    max_seq_len: int = 128       # 最大序列长度
    n_layers: int = 2            # Transformer 层数
    n_heads: int = 4             # 注意力头数
    d_model: int = 128           # 隐藏层维度
    d_ff: int = 512              # FFN 中间层维度
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # 预训练
    pretrain_epochs: int = 30
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 8
    # SFT
    sft_epochs: int = 40
    sft_lr: float = 5e-4
    sft_batch_size: int = 4
    # DPO
    dpo_epochs: int = 10
    dpo_lr: float = 1e-4
    dpo_batch_size: int = 4
    dpo_beta: float = 0.1        # DPO 温度参数


# ============================================================================
# 阶段 1：数据准备 (Data Preparation)
# ============================================================================

print("=" * 70)
print("阶段 1：数据准备 (Data Preparation)")
print("=" * 70)

# --- 1a. 原始数据收集与清洗 ---

RAW_CORPUS = """
<html><body>The sun rises in the east and sets in the west.</body></html>
Water freezes at zero degrees and boils at one hundred degrees.
  \t  The earth revolves around the sun in about 365 days.   \n\n
<p>Light travels faster than sound through empty space.</p>
Birds can fly because they have hollow bones and wings.
<div class="ad">BUY NOW! BEST DEALS!</div>
Fish breathe through gills to extract oxygen from water.
Plants convert sunlight into energy through photosynthesis.
The moon orbits the earth approximately once every 28 days.
<script>alert('spam')</script>
Mountains are formed by the movement of tectonic plates.
Rivers flow from higher elevations toward the sea.
"""

# 模拟中文语料
RAW_CORPUS_ZH = """
太阳从东方升起，从西方落下。
水在零度时结冰，在一百度时沸腾。
地球绕太阳公转一圈大约需要365天。
光的传播速度比声音快。
鸟类能飞行是因为它们有中空的骨骼和翅膀。
鱼通过鳃呼吸来从水中获取氧气。
植物通过光合作用将阳光转化为能量。
月球绕地球公转一圈大约需要28天。
"""


def clean_text(raw: str) -> list[str]:
    """模拟数据清洗：去除 HTML 标签、广告、脚本、空行。"""
    import re
    lines = raw.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 去除 HTML 标签
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if not line:
            continue
        # 过滤广告/垃圾内容
        if any(spam in line.upper() for spam in ["BUY NOW", "BEST DEALS", "ALERT"]):
            continue
        cleaned.append(line)
    return cleaned


print("\n[1a] 数据清洗 (Data Cleaning)")
clean_en = clean_text(RAW_CORPUS)
clean_zh = clean_text(RAW_CORPUS_ZH)
all_clean = clean_en + clean_zh
print(f"  原始语料行数: {len(RAW_CORPUS.strip().split(chr(10)))}")
print(f"  清洗后句子数: {len(all_clean)}")
for s in all_clean[:3]:
    print(f"    -> {s}")
print(f"    ... (共 {len(all_clean)} 条)")


# --- 1b. Tokenization (字节级分词) ---

print("\n[1b] 分词 (Tokenization) — 字节级 (Byte-level)")
print("  模型不认识文字，需要将文本转为数字序列。")
print("  这里采用最简单的字节级编码：每个字节 (0-255) 就是一个 Token。")

# 特殊 Token
PAD_TOKEN = 0
BOS_TOKEN = 1   # <BOS> 句子开始
EOS_TOKEN = 2   # <EOS> 句子结束
SEP_TOKEN = 3   # <SEP> 分隔符（用于 SFT 的 prompt/response 分隔）


def encode(text: str) -> list[int]:
    """将文本编码为字节级 Token ID 列表（偏移 +4 以避开特殊 Token）。"""
    return [min(b + 4, 255) for b in text.encode("utf-8")]


def decode(tokens: list[int]) -> str:
    """将 Token ID 列表解码回文本。"""
    bytes_list = []
    for t in tokens:
        if t <= 3:
            continue  # 跳过特殊 Token
        bytes_list.append(max(t - 4, 0))
    return bytes(bytes_list).decode("utf-8", errors="replace")


# 演示
demo_text = "Hello 你好"
demo_tokens = encode(demo_text)
print(f"\n  示例: \"{demo_text}\"")
print(f"  -> Token IDs: {demo_tokens}")
print(f"  -> 解码回来: \"{decode(demo_tokens)}\"")
print(f"  -> 词表大小: 256 (一个字节的所有可能值)")


# ============================================================================
# 模型定义：微型 GPT (Mini GPT)
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制 (Multi-Head Self-Attention)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 计算 Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆分多头
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数 (Scaled Dot-Product Attention)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # 因果掩码 (Causal Mask)：只能看到前面的 Token，不能偷看后面的
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """前馈网络 (Feed-Forward Network)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个 Transformer Decoder Block = 自注意力 + 前馈网络"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # 残差连接
        x = x + self.ff(self.ln2(x))     # 残差连接
        return x


class MiniGPT(nn.Module):
    """
    微型 GPT 模型
    - 字节级词表 (vocab_size=256)
    - 2 层 Transformer Decoder
    - ~5M 参数
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定 (Weight Tying)：输入 Embedding 和输出 Head 共享权重
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过最大值 {self.config.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_TOKEN,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.8) -> torch.Tensor:
        """自回归生成 (Autoregressive Generation)"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == EOS_TOKEN:
                break
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# 实例化模型并打印参数量
model_config = ModelConfig()
train_config = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniGPT(model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n模型已创建: MiniGPT")
print(f"  层数: {model_config.n_layers}, 注意力头: {model_config.n_heads}")
print(f"  隐藏维度: {model_config.d_model}, FFN 维度: {model_config.d_ff}")
print(f"  总参数量: {n_params:,} (~{n_params/1e6:.1f}M)")
print(f"  运行设备: {device}")


# ============================================================================
# 阶段 2：预训练 (Pre-training) — Next-Token Prediction
# ============================================================================

print("\n" + "=" * 70)
print("阶段 2：预训练 (Pre-training) — 文字接龙 (Next-Token Prediction)")
print("=" * 70)
print("核心思想：输入 [T1, T2, ..., Tn]，让模型预测 [T2, T3, ..., Tn+1]")
print("预测错了就通过反向传播调整权重，预测对了就继续。\n")


class PretrainDataset(Dataset):
    """预训练数据集：将清洗后的文本拼接，切成固定长度的训练样本。"""

    def __init__(self, sentences: list[str], seq_len: int, repeat: int = 10):
        # 将所有句子重复多次并拼接成一个长 Token 序列（小数据集需要重复）
        all_tokens = []
        for _ in range(repeat):
            random.shuffle(sentences)
            for s in sentences:
                all_tokens.extend([BOS_TOKEN] + encode(s) + [EOS_TOKEN])

        # 切成等长的训练样本
        self.samples = []
        for i in range(0, len(all_tokens) - seq_len - 1, seq_len // 2):
            chunk = all_tokens[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


pretrain_ds = PretrainDataset(all_clean, model_config.max_seq_len)
pretrain_dl = DataLoader(
    pretrain_ds, batch_size=train_config.pretrain_batch_size, shuffle=True
)
print(f"预训练样本数: {len(pretrain_ds)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.pretrain_lr)

for epoch in range(train_config.pretrain_epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    for x, y in pretrain_dl:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{train_config.pretrain_epochs}  |  Loss: {avg_loss:.4f}")

# 测试预训练后的生成（此时应该是"复读机"式输出）
print("\n预训练后的生成测试（此时模型是'复读机'，只会续写文本）：")
prompt = "The sun"
prompt_ids = torch.tensor([[BOS_TOKEN] + encode(prompt)], dtype=torch.long).to(device)
model.eval()
output = model.generate(prompt_ids, max_new_tokens=60, temperature=0.8)
generated = decode(output[0].tolist())
print(f"  输入: \"{prompt}\"")
print(f"  输出: \"{generated}\"")


# ============================================================================
# 阶段 3：指令微调 (Supervised Fine-Tuning, SFT)
# ============================================================================

print("\n" + "=" * 70)
print("阶段 3：指令微调 (Supervised Fine-Tuning)")
print("=" * 70)
print("目标：把'复读机'变成'问答助手'，学会 Prompt -> Response 的格式。\n")

# 构造 SFT 问答对数据
SFT_DATA = [
    {"prompt": "What rises in the east?",
     "response": "The sun rises in the east."},
    {"prompt": "At what temperature does water freeze?",
     "response": "Water freezes at zero degrees."},
    {"prompt": "How long does Earth take to orbit the sun?",
     "response": "The earth revolves around the sun in about 365 days."},
    {"prompt": "What travels faster, light or sound?",
     "response": "Light travels faster than sound."},
    {"prompt": "Why can birds fly?",
     "response": "Birds can fly because they have hollow bones and wings."},
    {"prompt": "How do fish breathe?",
     "response": "Fish breathe through gills to extract oxygen from water."},
    {"prompt": "What is photosynthesis?",
     "response": "Plants convert sunlight into energy through photosynthesis."},
    {"prompt": "How long does the moon take to orbit Earth?",
     "response": "The moon orbits the earth approximately once every 28 days."},
    {"prompt": "How are mountains formed?",
     "response": "Mountains are formed by the movement of tectonic plates."},
    {"prompt": "Where do rivers flow?",
     "response": "Rivers flow from higher elevations toward the sea."},
    # 中文问答对
    {"prompt": "太阳从哪里升起？",
     "response": "太阳从东方升起，从西方落下。"},
    {"prompt": "水在什么温度结冰？",
     "response": "水在零度时结冰，在一百度时沸腾。"},
    {"prompt": "光和声音哪个更快？",
     "response": "光的传播速度比声音快。"},
    {"prompt": "鱼是怎么呼吸的？",
     "response": "鱼通过鳃呼吸来从水中获取氧气。"},
]

print(f"SFT 训练对数: {len(SFT_DATA)}")
print(f"  示例: {json.dumps(SFT_DATA[0], ensure_ascii=False)}")


class SFTDataset(Dataset):
    """SFT 数据集：将问答对编码为 <BOS> prompt <SEP> response <EOS> 格式。"""

    def __init__(self, data: list[dict], max_len: int, repeat: int = 5):
        self.samples = []
        for item in data * repeat:  # 重复数据以增加训练量
            tokens = (
                [BOS_TOKEN]
                + encode(item["prompt"])
                + [SEP_TOKEN]
                + encode(item["response"])
                + [EOS_TOKEN]
            )
            if len(tokens) > max_len + 1:
                tokens = tokens[: max_len + 1]
            # Padding
            pad_len = max_len + 1 - len(tokens)
            tokens = tokens + [PAD_TOKEN] * pad_len
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


sft_ds = SFTDataset(SFT_DATA, model_config.max_seq_len)
sft_dl = DataLoader(sft_ds, batch_size=train_config.sft_batch_size, shuffle=True)

# SFT 使用更小的学习率
sft_optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.sft_lr)

for epoch in range(train_config.sft_epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    for x, y in sft_dl:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        sft_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{train_config.sft_epochs}  |  Loss: {avg_loss:.4f}")

# 测试 SFT 后的问答能力
print("\nSFT 后的问答测试（模型现在会尝试'回答'而非'续写'）：")
test_prompts = ["What rises in the east?", "How do fish breathe?", "太阳从哪里升起？"]
model.eval()
for p in test_prompts:
    prompt_ids = torch.tensor(
        [[BOS_TOKEN] + encode(p) + [SEP_TOKEN]], dtype=torch.long
    ).to(device)
    output = model.generate(prompt_ids, max_new_tokens=60, temperature=0.7)
    generated = decode(output[0].tolist())
    # 提取 response 部分
    if "\x03" in generated:  # SEP_TOKEN 解码后的字符
        generated = generated.split("\x03")[-1]
    print(f"  Q: {p}")
    print(f"  A: {generated.strip()}")
    print()


# ============================================================================
# 阶段 4：偏好对齐 (Alignment via DPO)
# ============================================================================

print("=" * 70)
print("阶段 4：偏好对齐 (Alignment via DPO — Direct Preference Optimization)")
print("=" * 70)
print("目标：让模型更倾向于生成'人类偏好'的高质量回答。")
print("方法：给定同一个问题的两个回答（好/差），让模型学会偏好'好回答'。\n")

# 构造偏好数据：chosen (好回答) vs rejected (差回答)
DPO_DATA = [
    {
        "prompt": "What rises in the east?",
        "chosen": "The sun rises in the east and sets in the west.",
        "rejected": "east east east sun sun",
    },
    {
        "prompt": "At what temperature does water freeze?",
        "chosen": "Water freezes at zero degrees Celsius.",
        "rejected": "Water is wet and cold sometimes.",
    },
    {
        "prompt": "Why can birds fly?",
        "chosen": "Birds can fly because they have hollow bones and wings that generate lift.",
        "rejected": "Birds fly. They just do.",
    },
    {
        "prompt": "How do fish breathe?",
        "chosen": "Fish breathe through gills to extract oxygen from water.",
        "rejected": "Fish live in water.",
    },
    {
        "prompt": "What is photosynthesis?",
        "chosen": "Plants convert sunlight into energy through photosynthesis.",
        "rejected": "Plants are green.",
    },
    {
        "prompt": "How are mountains formed?",
        "chosen": "Mountains are formed by the movement of tectonic plates over millions of years.",
        "rejected": "Mountains are big rocks.",
    },
    {
        "prompt": "太阳从哪里升起？",
        "chosen": "太阳从东方升起，从西方落下。",
        "rejected": "太阳太阳太阳",
    },
    {
        "prompt": "鱼是怎么呼吸的？",
        "chosen": "鱼通过鳃呼吸来从水中获取氧气。",
        "rejected": "鱼在水里。",
    },
]

print(f"DPO 偏好对数: {len(DPO_DATA)}")
print(f"  示例:")
print(f"    Prompt:   {DPO_DATA[0]['prompt']}")
print(f"    Chosen:   {DPO_DATA[0]['chosen']}")
print(f"    Rejected: {DPO_DATA[0]['rejected']}")


def encode_for_dpo(prompt: str, response: str, max_len: int) -> torch.Tensor:
    """将 prompt+response 编码为固定长度的 Token 序列。"""
    tokens = (
        [BOS_TOKEN] + encode(prompt) + [SEP_TOKEN] + encode(response) + [EOS_TOKEN]
    )
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    tokens = tokens + [PAD_TOKEN] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)


def compute_log_probs(
    model: MiniGPT, input_ids: torch.Tensor
) -> torch.Tensor:
    """计算序列的对数概率之和 (用于 DPO loss)。"""
    logits, _ = model(input_ids)
    # Shift: 用前 n-1 个 Token 的 logits 预测后 n-1 个 Token
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # 取每个位置对应 label 的 log_prob
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # 对非 PAD 位置求和
    mask = (shift_labels != PAD_TOKEN).float()
    return (token_log_probs * mask).sum(dim=-1)


# DPO 训练
# 先保存一份 SFT 后的模型作为 reference model (π_ref)
ref_model = MiniGPT(model_config).to(device)
ref_model.load_state_dict(model.state_dict())
ref_model.eval()

dpo_optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.dpo_lr)

for epoch in range(train_config.dpo_epochs):
    model.train()
    total_loss = 0
    random.shuffle(DPO_DATA)

    # 手动按 batch 处理
    for i in range(0, len(DPO_DATA), train_config.dpo_batch_size):
        batch = DPO_DATA[i : i + train_config.dpo_batch_size]

        chosen_ids = torch.stack(
            [encode_for_dpo(d["prompt"], d["chosen"], model_config.max_seq_len) for d in batch]
        ).to(device)
        rejected_ids = torch.stack(
            [encode_for_dpo(d["prompt"], d["rejected"], model_config.max_seq_len) for d in batch]
        ).to(device)

        # 当前策略模型的 log_prob
        pi_chosen = compute_log_probs(model, chosen_ids)
        pi_rejected = compute_log_probs(model, rejected_ids)

        # 参考模型的 log_prob (不参与梯度计算)
        with torch.no_grad():
            ref_chosen = compute_log_probs(ref_model, chosen_ids)
            ref_rejected = compute_log_probs(ref_model, rejected_ids)

        # DPO Loss = -log σ(β * [(log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))])
        chosen_reward = pi_chosen - ref_chosen
        rejected_reward = pi_rejected - ref_rejected
        loss = -F.logsigmoid(train_config.dpo_beta * (chosen_reward - rejected_reward)).mean()

        dpo_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        dpo_optimizer.step()
        total_loss += loss.item()

    n_steps = max(len(DPO_DATA) // train_config.dpo_batch_size, 1)
    avg_loss = total_loss / n_steps
    print(f"  Epoch {epoch+1}/{train_config.dpo_epochs}  |  DPO Loss: {avg_loss:.4f}")

# 对齐后的最终测试
print("\n对齐后的最终生成测试：")
model.eval()
final_test = [
    "What rises in the east?",
    "How do fish breathe?",
    "Why can birds fly?",
    "太阳从哪里升起？",
    "鱼是怎么呼吸的？",
]
for p in final_test:
    prompt_ids = torch.tensor(
        [[BOS_TOKEN] + encode(p) + [SEP_TOKEN]], dtype=torch.long
    ).to(device)
    output = model.generate(prompt_ids, max_new_tokens=80, temperature=0.7)
    generated = decode(output[0].tolist())
    if "\x03" in generated:
        generated = generated.split("\x03")[-1]
    print(f"  Q: {p}")
    print(f"  A: {generated.strip()}")
    print()


# ============================================================================
# 总结
# ============================================================================

print("=" * 70)
print("训练完成！回顾四个阶段：")
print("=" * 70)
print("""
  1. 数据准备    → 清洗原始文本 + 字节级 Tokenization
  2. 预训练      → Next-Token Prediction（文字接龙），得到"基座模型"
  3. 指令微调    → 用问答对训练，让模型学会"回答"而非"续写"
  4. 偏好对齐    → DPO 让模型偏好高质量回答，拒绝低质量回答

关键概念对照：
  ┌──────────────────┬────────────────────────────────────┐
  │  大模型概念       │  软件工程类比                       │
  ├──────────────────┼────────────────────────────────────┤
  │  预训练          │  编译操作系统内核                    │
  │  Token           │  字节码 (Bytecode)                  │
  │  权重 (Weights)  │  编译产物（二进制文件）               │
  │  SFT 微调        │  在内核上封装 API 接口               │
  │  LoRA            │  热补丁 / 插件系统                   │
  │  DPO 对齐        │  CI/CD Linter + 代码规范检查         │
  │  反向传播        │  编译器的错误反馈 → 修改源码          │
  └──────────────────┴────────────────────────────────────┘

模型参数量: """ + f"{n_params:,}" + """ (玩具级)
实际大模型:  7B ~ 405B 参数 (工业级)
""")

# 保存模型权重
save_path = "toy_gpt_final.pt"
torch.save(model.state_dict(), save_path)
print(f"模型权重已保存到: {save_path}")
print(f"文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")
print("\nDone! 🎉")
