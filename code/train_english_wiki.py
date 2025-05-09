"""
Distill BERT-base to DistilBERT **using English Wikipedia only** 
(un-supervised; no SST-2, no hard CE loss).

Run:
    python distill_bert_wiki.py
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
from datasets import load_dataset, DatasetDict, load_from_disk

# ---------- 1. Config -------------------------------------------------- #
TEACHER_MODEL_NAME = "bert-base-uncased"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
OUTPUT_DIR = "./bert_wiki"
EPOCHS     = 1
BS_TRAIN   = 32
BS_EVAL    = 64
LR         = 5e-5
TEMPERATURE    = 2.0
ALPHA_SOFT     = 1.0
ALPHA_HARD     = 0.0          # <<< CHANGED – disable CE (no gold labels)
ALPHA_COS      = 2.0
SEED       = 42

SPLIT_TEST = 0.1
SPLIT_VAL  = 0.1
WIKI_DUMP_ID = "20220301.en"  # <<< ADDED  – HF Wikipedia snapshot
NUM_PROC = 16 # Modify to fit your cpu core numbers. Make sure to leave 4-6 cores for your system.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. Load & split Wikipedia --------------------------------- #
# Run the first time. Then you can load from disk.
def load_wiki_from_remote():
    print("\nLoading English Wikipedia dump …")                               # <<< CHANGED
    # Data is downloaded to C:\Users\<YourUsername>\.cache\huggingface\datasets\
    wiki = load_dataset("wikipedia", WIKI_DUMP_ID, split="train",trust_remote_code=True)
    wiki = wiki.shuffle(seed=SEED)
    # optional quick-run cap
    # wiki = wiki.select(range(min(len(wiki), 1_000_000)))                      # <<< ADDED

    def keep_text(example):                                                   # <<< ADDED
        return {"sentence": example["text"]}                                  # <<< ADDED

    wiki = wiki.map(keep_text, remove_columns=wiki.column_names, num_proc=NUM_PROC)    
    wiki.save_to_disk("./cached/wiki_kept_text")

def load_wiki_mapped():
    wiki = load_from_disk("./cached/wiki_kept_text")
    return wiki

def split_and_tokenize_dataset(wiki):
    split1 = wiki.train_test_split(test_size=SPLIT_TEST, seed=SEED)
    train_val = split1["train"]
    test_ds   = split1["test"]

    val_ratio_relative = SPLIT_VAL / (1.0 - SPLIT_TEST)
    split2 = train_val.train_test_split(test_size=val_ratio_relative, seed=SEED)
    train_ds = split2["train"]
    val_ds   = split2["test"]

    print(f"Wiki split sizes — train: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}")                        # <<< CHANGED

    # ---------- 3. Tokenisation ------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)

    def tokenize(batch: Dict):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_token_type_ids=False
        )

    ds = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    ds = ds.map(tokenize, batched=True, remove_columns=["sentence"], num_proc=NUM_PROC)          # <<< CHANGED
    ds.set_format(type="torch")
    ds.save_to_disk("./cached/wiki_tokenized")


# # ---------- 4. Teacher & student -------------------------------------- #
def create_models():
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

    student_config = DistilBertConfig(
        vocab_size     = 30522,
        max_position_embeddings = 512,
        n_layers       = 6,
        n_heads        = 12,
        dim            = 768,
        hidden_dim     = 3072,
        dropout        = 0.1,
        attention_dropout = 0.1,
        classifier_dropout = 0.1,
        num_labels     = 2,
    )
    student = DistilBertForSequenceClassification(student_config)
    return teacher, student

# # ---------- 5. Distillation Trainer ----------------------------------- #
class DistillationTrainer(Trainer):
    """
    Unsupervised distillation on Wikipedia:
      • KL divergence on logits
      • Cosine distance on [CLS] embeddings
    """
    def __init__(self, teacher, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # <<< CHANGED – no label popping, unsupervised
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
            teacher_logits  = teacher_outputs.logits
            teacher_cls     = teacher_outputs.hidden_states[-1][:, 0]

        student_outputs = model(**inputs, output_hidden_states=True)
        student_logits  = student_outputs.logits
        student_cls     = student_outputs.hidden_states[-1][:, 0]

        loss_soft = F.kl_div(
            F.log_softmax(student_logits / TEMPERATURE, dim=-1),
            F.softmax   (teacher_logits / TEMPERATURE, dim=-1),
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)

        loss_cosine = (1.0 - F.cosine_similarity(student_cls, teacher_cls, dim=-1)).mean()

        total_loss = ALPHA_SOFT * loss_soft + ALPHA_COS * loss_cosine      # <<< CHANGED
        return (total_loss, student_outputs) if return_outputs else total_loss

# ---------- 6. TrainingArguments -------------------------------------- #
def training_arguments():
    return TrainingArguments(
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
        report_to=[],
    )

# # ---------- 7. Train --------------------------------------------------- #
def train_model(student,teacher,training_args,ds):
    print("\nStarting Wikipedia distillation …")                              # <<< CHANGED
    trainer = DistillationTrainer(
        model=student,
        teacher=teacher,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=None,                                                 # <<< CHANGED – no accuracy
    )

    trainer.train()
    trainer.save_model()
    return trainer
# print("Training complete.\nBest model saved to:", Path(OUTPUT_DIR).resolve())

# # ---------- 8. Evaluate ------------------------------------------------ #
def evaluate(trainer,ds):
    print("\nEvaluating on held-out Wikipedia test split (loss only) …")       # <<< CHANGED
    test_metrics = trainer.evaluate(ds["test"])
    print(f"Test KL+Cosine loss: {test_metrics['eval_loss']:.4f}")             # <<< CHANGED


if __name__ == "__main__":
    # Only run the first time to preprocess and save
    text_extracted_path = Path("./cached/wiki_kept_text")
    tokenized_path = Path("./cached/wiki_tokenized")
    if not tokenized_path.exists():
        if not text_extracted_path.exists():
            load_wiki_from_remote()
        
        print("Tokenizing dataset …")
        wiki = load_wiki_mapped()
        print(f"Loaded preprocessed dataset with {len(wiki)} articles.")
        split_and_tokenize_dataset(wiki)
    
    print("Loading cached tokenized dataset …")
    ds = load_from_disk(tokenized_path)
    teacher, student = create_models()
    train_args = training_arguments()
    trainer = train_model(student,teacher,train_args,ds)
    evaluate(trainer,ds)
