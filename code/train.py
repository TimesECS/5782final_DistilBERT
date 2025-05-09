"""
Distill BERT‑base on SST‑2, but first recombine the original GLUE
train + validation sets and create fresh train / val / test splits.

Optimised for a single 12 GB RTX 4070 (≈ 6–8 h with fp16).

Run:
    python distill_bert_sst2.py
"""
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DistilBertConfig
)
from datasets import load_dataset, DatasetDict
import evaluate  

# ---------- 1. Config -------------------------------------------------- #
TEACHER_MODEL_NAME = "bert-base-uncased"       # swap for an SST‑2‑finetuned ckpt for a stronger teacher
STUDENT_MODEL_NAME = "distilbert-base-uncased"
TASK_NAME  = "sst2"
MAX_LEN    = 128
OUTPUT_DIR = "./bert_sst2_minimal_dataset"
EPOCHS     = 1
BS_TRAIN   = 32
BS_EVAL    = 64
LR         = 5e-5
TEMPERATURE    = 2.0
ALPHA_SOFT     = 1.0
ALPHA_HARD     = 5.0
ALPHA_COS = 2.0
SEED       = 42

SPLIT_TEST = 0.1
SPLIT_VAL  = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Build fresh splits ------------------------------------- #
print("\nLoading GLUE/SST‑2 and creating custom splits …")
# concat original train + validation (both have labels)
combined = load_dataset("glue", TASK_NAME, split="train+validation")  # ~ 67 k examples
combined = combined.shuffle(seed=SEED)

# first, carve out test
split1 = combined.train_test_split(test_size=SPLIT_TEST, seed=SEED)
train_val = split1["train"]
test_ds   = split1["test"]

# now carve validation out of the remaining data
val_ratio_relative = SPLIT_VAL / (1.0 - SPLIT_TEST)
split2 = train_val.train_test_split(test_size=val_ratio_relative, seed=SEED)
train_ds = split2["train"]
val_ds   = split2["test"]

print(f"New split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)

def tokenize(batch: Dict):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_token_type_ids=False
    )

# create a DatasetDict so we can tokenise in one go
ds = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

print(ds["train"][0])
print(ds["validation"][0])
print(ds["test"][0])

ds = ds.map(
    tokenize,
    batched=True,
    remove_columns=["sentence", "idx"],   # drop raw text / id columns
)
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch")

# ---------- 3. Load teacher & student ---------------------------------- #
print("Loading teacher and student models …")
teacher = BertForSequenceClassification.from_pretrained(
    TEACHER_MODEL_NAME, 
    num_labels=2,
    output_hidden_states=True,               
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student_config = DistilBertConfig(           # identical to distilbert‑base
    vocab_size     = tokenizer.vocab_size,   # 30 522  (same as BERT/DistilBERT)
    max_position_embeddings = 512,
    n_layers       = 6,
    n_heads        = 12,
    dim            = 768,
    hidden_dim     = 3072,
    dropout        = 0.1,
    attention_dropout = 0.1,
    classifier_dropout = 0.1,
    num_labels     = 2,                      # ← task‑specific
)

student = DistilBertForSequenceClassification(student_config)

# ---------- 4. Metric --------------------------------------------------- #
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

# ---------- 5. Distillation Trainer ------------------------------------ #
class DistillationTrainer(Trainer):
    """
    Custom Trainer that combines
      • soft KL loss w.r.t. teacher logits
      • hard CE loss w.r.t. true labels
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        labels = inputs.pop("labels")

        if model.training:
            with torch.no_grad():
                teacher_outputs = teacher(**inputs, output_hidden_states=True)
                teacher_logits = teacher_outputs.logits
                teacher_cls     = teacher_outputs.hidden_states[-1][:, 0] 

            # forward (student)
            student_outputs = model(**inputs, output_hidden_states = True)
            student_logits = student_outputs.logits

            # hard targets
            loss_hard = F.cross_entropy(student_logits, labels)

            # soft targets
            loss_soft = F.kl_div(
                F.log_softmax(student_logits / TEMPERATURE, dim=-1),
                F.softmax(teacher_logits / TEMPERATURE, dim=-1),
                reduction="batchmean",
            ) * (TEMPERATURE ** 2)

            # Cosine-distance loss
            student_cls = student_outputs.hidden_states[-1][:,0]
            loss_cosine = (1.0 - F.cosine_similarity(student_cls, teacher_cls, dim=-1)).mean()

            total_loss = (
                ALPHA_SOFT * loss_soft +
                ALPHA_HARD * loss_hard +
                ALPHA_COS  * loss_cosine              
            )
        else:
            with torch.no_grad():
                student_outputs = model(**inputs)
                student_logits = student_outputs.logits

                # hard targets
                total_loss = F.cross_entropy(student_logits, labels)

        return (total_loss, student_outputs) if return_outputs else total_loss

# ---------- 6. TrainingArguments --------------------------------------- #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BS_TRAIN,
    per_device_eval_batch_size=BS_EVAL,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    seed=SEED,
    report_to=[],             # disable external loggers (wandb, etc.)
)

# ---------- 7. Train ---------------------------------------------------- #
print("\nStarting distillation …")
trainer = DistillationTrainer(
    model=student,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
print("Training complete.\nBest model saved to:", Path(OUTPUT_DIR).resolve())

# ---------- 8. Evaluate on fresh test set ------------------------------ #
print("\nEvaluating on the fresh held‑out *test* split …")
test_metrics = trainer.evaluate(ds["test"])
print(f"Test accuracy: {test_metrics['eval_accuracy']:.4%}")