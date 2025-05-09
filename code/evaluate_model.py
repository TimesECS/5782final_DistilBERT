from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# ----------------------------- 1. Config ----------------------------------- #
MODEL_DIR  = "bert_sst2_minimal_dataset/checkpoint-1706"     # ← your saved student
OUT_DIR    = Path("./bert_sst2_minimal_dataset")
MAX_LEN    = 128
BATCH_SIZE = 32
SAMPLES    = 5_000                       # cap per‑task examples
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42
original_performance = {
    "CoLA": 51.3,
    "MRPC": 87.5,
    "QQP": 88.5,
    "RTE": 59.9,
    "SST-2": 91.3
}

# Nine core GLUE tasks → columns that contain the text fields
GLUE_TASKS: Dict[str, Tuple[str, str | None]] = {
    "cola" : ("sentence",       None),
    "sst2" : ("sentence",       None),
    "mrpc" : ("sentence1",      "sentence2"),
    "qqp"  : ("question1",      "question2"),
    "rte"  : ("sentence1",      "sentence2")
}

random.seed(SEED)
np.random.seed(SEED)

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_DIR)
    .eval()
    .to(DEVICE)
)

collator = DataCollatorWithPadding(tok, return_tensors="pt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

scores: Dict[str, float] = {}

# ---------------------- 2. Helper: prepare a task -------------------------- #

def prepare_loader(task: str) -> DataLoader:
    """Load ≤5 000 rows from the *train* split of *task* and return a DataLoader."""
    text1_col, text2_col = GLUE_TASKS[task]
    raw = load_dataset("glue", task, split="train")
    if len(raw) > SAMPLES:
        raw = raw.shuffle(seed=SEED).select(range(SAMPLES))

    def _encode(batch):
        if text2_col is None:
            enc = tok(
                batch[text1_col],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
        else:
            enc = tok(
                batch[text1_col],
                batch[text2_col],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
        enc.pop("token_type_ids", None)
        if "label" in batch:
            enc["labels"] = batch["label"]
        return enc

    tokenised = raw.map(_encode, batched=True, remove_columns=raw.column_names)
    return DataLoader(
        tokenised,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        pin_memory=(DEVICE == "cuda"),
    )

# ---------------------- 3. Evaluate one task ------------------------------ #

def evaluate_task(task: str) -> float:
    loader = prepare_loader(task)
    preds: List[float] = []
    golds: List[float] = []

    for batch in loader:
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            logits = model(**batch_gpu).logits

        if task == "stsb":
            # Regression – logits shape [B, 1]
            preds.extend(logits.squeeze(-1).cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())
        else:
            probs = logits.softmax(dim=-1)
            preds.extend(probs.argmax(dim=-1).cpu().tolist())
            golds.extend(batch["labels"].cpu().tolist())

    if task == "stsb":
        score = 1.0 - mean_squared_error(golds, preds)  # pseudo‑accuracy (higher is better)
    else:
        # Some datasets have labels outside the model's output range (e.g. MNLI → label 2).
        # We ignore those samples when computing accuracy.
        mask = [g in {0, 1} for g in golds]
        if not any(mask):
            score = float("nan")
        else:
            score = accuracy_score([g for g, m in zip(golds, mask) if m], [p for p, m in zip(preds, mask) if m])
    return score

# --------------------------- 4. Main loop --------------------------------- #
for task in GLUE_TASKS:
    try:
        score = evaluate_task(task)
        scores[task.upper()] = score
        print(f"{task.upper():5s}: {score:.4f}")
    except Exception as e:
        print(f"{task.upper():5s}: ERROR → {e}")
        scores[task.upper()] = float("nan")

# --------------------------- 5. Plot -------------------------------------- #
plt.figure(figsize=(9, 4))

# 1) Make keys consistent (e.g. "SST-2" -> "SST2") and convert to 0‑1 range
orig_norm = {
    k.replace("-", "").upper(): v / 100.0
    for k, v in original_performance.items()
}
labels = list(scores.keys())                       # e.g. ['COLA', 'SST2', ...]
student_vals = [scores[t] for t in labels]
bert_vals    = [orig_norm.get(t, np.nan) for t in labels]

# 2) Grouped bars
x      = np.arange(len(labels))
width  = 0.35
plt.bar(x - width/2, student_vals, width, label="Our Distil")
plt.bar(x + width/2, bert_vals,    width, label="Original Distil")

# 3) Aesthetics
plt.xticks(x, labels)
plt.ylabel("Accuracy (train split)")
plt.title("Zero‑shot GLUE performance (≤5 000 samples)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plot_path = OUT_DIR / "glue_train_accuracy_vs_bert.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print("Figure with comparison saved to", plot_path)

# """
# Evaluate a distilled DistilBERT checkpoint on the SST‑2 *test* (or any) split.

# Saved artefacts
# ---------------
# • <split>_predictions.txt   – one label per line  
# • <split>_predictions.csv   – idx, pred_label, prob_neg, prob_pos  
# • pred_distribution.png     – bar chart of class counts  
# The script also prints overall accuracy **if** the split actually has gold
# labels (GLUE SST‑2’s public *test* split does *not* – its labels are ‑1).

# Run
# ---
#     python evaluate_distilbert_sst2.py

# Dependencies
# ------------
#     pip install torch transformers datasets matplotlib
# """
# from pathlib import Path
# import csv
# from typing import List, Tuple

# import torch
# from torch.utils.data import DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     DataCollatorWithPadding,
# )
# from datasets import load_dataset
# import matplotlib.pyplot as plt

# # ---------- 1. Config -------------------------------------------------- #
# TASK_NAME  = "sst2"
# MODEL_DIR  = "./bert_sst2_distilled/checkpoint-3411"   # your saved student
# MAX_LEN    = 128
# SPLIT      = "test"                                    # "train", "validation" or "test"
# OUT_DIR    = Path("./bert_sst2_distilled")             # where files will be written
# DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 32

# # ---------- 2. Load model & tokenizer ---------------------------------- #
# tok   = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = (AutoModelForSequenceClassification
#          .from_pretrained(MODEL_DIR)
#          .eval()
#          .to(DEVICE))

# # ---------- 3. Dataset & tokenisation ---------------------------------- #
# raw_ds = load_dataset("glue", TASK_NAME, split=SPLIT)

# def _encode(batch):
#     enc = tok(
#         batch["sentence"],
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LEN,
#     )
#     enc.pop("token_type_ids", None)        # DistilBERT doesn't use them
#     # keep label → rename to "labels" (needed for accuracy)
#     if "label" in batch:
#         enc["labels"] = batch["label"]
#     return enc

# tokenised = raw_ds.map(
#     _encode,
#     batched=True,
#     remove_columns=["sentence", "idx"]     # keep label column
# )

# collator = DataCollatorWithPadding(tok, return_tensors="pt")
# loader   = DataLoader(
#     tokenised,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collator,
#     pin_memory=(DEVICE == "cuda"),
# )

# # ---------- 4. Inference loop ------------------------------------------ #
# OUT_DIR.mkdir(exist_ok=True)
# txt_path  = OUT_DIR / f"{SPLIT}_predictions.txt"
# csv_path  = OUT_DIR / f"{SPLIT}_predictions.csv"

# all_labels: List[int]               = []
# all_probs:  List[Tuple[float,float]] = []

# correct = 0        # for accuracy
# total   = 0
# has_valid_labels = False  # flips True if we encounter any label ≥ 0

# with txt_path.open("w") as txt_f, csv_path.open("w", newline="") as csv_f:
#     writer = csv.writer(csv_f)
#     writer.writerow(["idx", "pred_label", "prob_neg", "prob_pos"])

#     idx_offset = 0
#     for batch in loader:
#         batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()
#                      if k in {"input_ids", "attention_mask"}}

#         with torch.no_grad():
#             logits = model(**batch_gpu).logits
#             probs  = logits.softmax(dim=-1)

#         preds     = probs.argmax(dim=-1).cpu()
#         probs_cpu = probs.cpu()
#         gold      = batch.get("labels")

#         # --- accuracy bookkeeping (skip labels == -1) ------------------ #
#         if gold is not None:
#             mask = gold >= 0        # GLUE test labels are -1
#             if mask.any():
#                 has_valid_labels = True
#                 correct += (preds[mask] == gold[mask]).sum().item()
#                 total   += mask.sum().item()

#         # --- accumulate for later plotting ----------------------------- #
#         all_labels.extend(preds.tolist())
#         all_probs.extend(probs_cpu.tolist())

#         # --- write TXT / CSV ------------------------------------------- #
#         for i in range(preds.size(0)):
#             lbl, pr = int(preds[i]), probs_cpu[i].tolist()
#             txt_f.write(f"{lbl}\n")
#             writer.writerow([idx_offset + i, lbl, pr[0], pr[1]])

#         idx_offset += preds.size(0)

# print(f"Predictions saved to\n  {txt_path}\n  {csv_path}")

# if has_valid_labels:
#     accuracy = correct / total
#     print(f"Accuracy on *{SPLIT}* split: {accuracy:.4%}")
# else:
#     print(f"No gold labels available in the *{SPLIT}* split – accuracy not computed.")

# # ---------- 5. Plot distribution --------------------------------------- #
# pos = sum(all_labels)
# neg = len(all_labels) - pos

# plt.figure(figsize=(4, 3))
# plt.bar(["negative (0)", "positive (1)"], [neg, pos], width=0.6)
# plt.title(f"Prediction distribution on SST‑2 {SPLIT} split")
# plt.ylabel("Count")
# plt.tight_layout()

# plot_path = OUT_DIR / "pred_distribution.png"
# plt.savefig(plot_path)
# plt.close()
# print("Bar chart saved to", plot_path)

# tsv_path = OUT_DIR / "SST-2.tsv"          # exact filename matters!
# with tsv_path.open("w") as f:
#     f.write("index\tprediction\n")        # header row
#     for idx, pred in enumerate(all_labels):
#         f.write(f"{idx}\t{pred}\n")
# print("Submission file written to", tsv_path)
