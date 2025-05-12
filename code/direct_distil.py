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

TEACHER_MODEL_NAME = "bert-base-uncased"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
TASKS = ["sst2","mrpc","qqp","rte","cola"]
TASK2COLS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp" : ("question1", "question2"),
    "cola": ("sentence", None),
    "rte" : ("sentence1", "sentence2"),
}
MAX_LEN    = 128
EPOCHS     = 3
BS_TRAIN   = 64
BS_EVAL    = 64
LR         = 5e-5
TEMPERATURE    = 2.0
ALPHA_SOFT     = 1.0
ALPHA_HARD     = 5.0
ALPHA_COS = 2.0
SEED       = 1
SPLIT_TEST = 0.1
SPLIT_VAL  = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("accuracy")

def load_ds_and_tokenize(task,sent1_key,sent2_key):
    print("Loading GLUE/" + task + " and creating custom splits …")
    combined = load_dataset("glue", task, split="train+validation")
    combined = combined.shuffle(seed=SEED)
    split1    = combined.train_test_split(test_size=SPLIT_TEST, seed=SEED)
    train_val = split1["train"]
    test_ds   = split1["test"]
    val_ratio_relative = SPLIT_VAL / (1.0 - SPLIT_TEST)
    split2   = train_val.train_test_split(test_size=val_ratio_relative, seed=SEED)
    train_ds = split2["train"]
    val_ds   = split2["test"]

    print(
        f"New split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, use_fast=True)
    def tokenize(batch: Dict):
        if sent2_key is None:
            return tokenizer(
                batch[sent1_key],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_token_type_ids=False,
            )
        else:
            return tokenizer(
                batch[sent1_key],
                batch[sent2_key],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_token_type_ids=False,
            )
    ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    remove_cols = [sent1_key, "idx"] + ([sent2_key] if sent2_key else []) 
    ds = ds.map(tokenize, batched=True, remove_columns=remove_cols)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")
    return ds

def load_models():
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

    student = DistilBertForSequenceClassification(student_config).to(device)
    return teacher, student

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

class DistillationTrainer(Trainer):
    """
    Custom Trainer that combines
      • soft KL loss w.r.t. teacher logits
      • hard CE loss w.r.t. true labels
    """
    def __init__(self, teacher, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        labels = inputs.pop("labels")

        if model.training:
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
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

def main():
    for task in TASKS:
        OUTPUT_DIR = f"./checkpoints_direct_distil/{task}"
        sent1_key, sent2_key = TASK2COLS[task]
        ds = load_ds_and_tokenize(task,sent1_key,sent2_key)
        teacher, student = load_models()
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
            report_to=[],
        )
        print("\nStarting distillation …")
        trainer = DistillationTrainer(
            model=student,
            teacher=teacher,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model()
        print("Training complete.\nBest model saved to:", Path(OUTPUT_DIR).resolve())

        print("\nEvaluating on the fresh held‑out *test* split …")
        test_metrics = trainer.evaluate(ds["test"])
        print(f"Test accuracy: {test_metrics['eval_accuracy']:.4%}")

if __name__ == "__main__":
    main()
