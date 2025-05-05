#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py  --  IMDb 任务级蒸馏
Teacher : google/bert_uncased_L-4_H-512_A-8
Student : 2 layers / 256 hidden / 4 heads
单卡 GPU 即可跑通（~2‑3h）
"""

import torch, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertConfig, BertForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score
from functools import partial
import argparse, os, random

# ---------- 超参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="./distil_imdb", type=str)
parser.add_argument("--epochs",      default=3, type=int)
parser.add_argument("--bs_train",    default=16, type=int)
parser.add_argument("--bs_eval",     default=32, type=int)
parser.add_argument("--lr",          default=2e-5, type=float)
parser.add_argument("--max_len",     default=256, type=int)
parser.add_argument("--alpha",       default=0.7, type=float,  # KD loss 权重
                    help="Proportion of KD loss (1-alpha is CE loss)")
parser.add_argument("--seed",        default=42, type=int)
args = parser.parse_args()

# ---------- 随机种子 ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. 加载 IMDb ----------
print("▶ Loading IMDb …")
imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(
    "google/bert_uncased_L-4_H-512_A-8",
    use_fast=True
)

def tokenize(batch, max_len):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len
    )

imdb = imdb.map(partial(tokenize, max_len=args.max_len),
                batched=True, remove_columns=["text"])
imdb = imdb.rename_column("label", "labels")
imdb.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------- 2. Teacher ----------
print("▶ Loading teacher model …")
teacher = AutoModelForSequenceClassification.from_pretrained(
    "google/bert_uncased_L-4_H-512_A-8",
    num_labels=2
).to(device)
teacher.eval()        # 推理模式
for p in teacher.parameters():
    p.requires_grad = False

# ---------- 3. Student ----------
print("▶ Building student config …")
stu_cfg = BertConfig(
    vocab_size     = tokenizer.vocab_size,
    hidden_size    = 256,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    intermediate_size   = 1024,
    pad_token_id   = tokenizer.pad_token_id,
    num_labels     = 2
)
student = BertForSequenceClassification(stu_cfg)
# （可选）把 teacher 前两层权重拷给 student 以加速收敛
with torch.no_grad():
    for i in range(2):
        student.bert.encoder.layer[i].load_state_dict(
            teacher.bert.encoder.layer[i].state_dict(), strict=False
        )
student.to(device)

# ---------- 4. 蒸馏损失 ----------
def distillation_loss(student_logits, teacher_logits, labels,
                      T=2.0, alpha=args.alpha):
    """
    total_loss = alpha * KD + (1-alpha) * CE
    """
    kd_loss = torch.nn.functional.kl_div(
        torch.log_softmax(student_logits / T, dim=-1),
        torch.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean") * (T * T)
    ce_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

# ---------- 5. 自定义 Trainer ----------
class DistilBertTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        # Teacher 输出
        with torch.no_grad():
            t_out = teacher(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device)
            )
        # Student 前向
        s_out = model(**inputs)
        loss = distillation_loss(
            s_out.logits, t_out.logits.detach(), labels.to(device)
        )
        return (loss, s_out) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ---------- 6. TrainingArguments ----------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.bs_train,
    per_device_eval_batch_size=args.bs_eval,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none",
    save_total_limit=1,
    seed=args.seed
)

# ---------- 7. 开始训练 ----------
trainer = DistilBertTrainer(
    model=student,
    args=training_args,
    train_dataset=imdb["train"],
    eval_dataset=imdb["test"],
    compute_metrics=compute_metrics
)

print("▶ Training student with knowledge distillation …")
trainer.train()
eval_res = trainer.evaluate()
print(f"▶ Final accuracy: {eval_res['eval_accuracy']:.4f}")
print(f"▶ Final loss:      {eval_res['eval_loss']:.4f}")

# ---------- 8. 保存 ----------
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"✅ Model & tokenizer saved to {args.output_dir}")
