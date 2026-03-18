# Toy LLM Training Demo (玩具级大模型训练全流程)

在普通笔记本 CPU 上几分钟内跑通大模型训练的**四个核心阶段**：

| 阶段 | 名称 | 说明 |
|------|------|------|
| 1 | 数据准备 (Data Preparation) | 文本清洗 + 字节级 Tokenization |
| 2 | 预训练 (Pre-training) | Next-Token Prediction 文字接龙 |
| 3 | 指令微调 (SFT) | 用问答对让模型学会"回答"而非"续写" |
| 4 | 偏好对齐 (DPO) | 让模型偏好高质量回答 |

## 模型规格

- **架构**: 2 层 Transformer Decoder (微型 GPT)
- **参数量**: ~5M
- **词表**: 字节级 (256)
- **运行环境**: 纯 PyTorch，CPU 即可

## 快速开始

```bash
pip install -r requirements.txt
python toy_llm_train.py
```

整个流程在 CPU 上约 1-3 分钟即可完成。
