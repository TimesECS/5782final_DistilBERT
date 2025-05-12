import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset,DatasetDict
from transformers import BertTokenizerFast
from tqdm.auto import tqdm
from model import DistilBertForMaskedLM, DistilBertForSequenceClassification

SEED = 1
SPLIT_TEST=0.1
SPLIT_VAL=0.1
NUM_PROC = 16

class CustomClassDataset(Dataset):
    def __init__(self, ids, attention_masks, labels):
        self.ids = ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return {
            "source_ids": self.ids[idx],
            "source_mask": self.attention_masks[idx],
            "label":self.labels[idx].long()
        }
    
def evaluate(model, val_loader, device):
    model.eval()
    val_running_loss = correct = total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            out = model(
                batch["source_ids"].to(device),
                batch["source_mask"].to(device),
                batch["label"].to(device),
            )
            val_running_loss += out["loss"].item()
            correct += (out["preds"] == batch["label"].to(device)).sum().item()
            total += batch["label"].size(0)
    return val_running_loss/len(val_loader), correct/total

def train(model, train_loader, val_loader, epochs, optimizer, device):
    train_loss_arr = []
    val_loss_arr = []
    val_acc_arr = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            out = model(
                batch["source_ids"].to(device),
                batch["source_mask"].to(device),
                batch["label"].to(device),
            )
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_running_loss, val_acc = evaluate(model, val_loader, device)
        running_loss /= len(train_loader)
        train_loss_arr.append(running_loss)
        val_loss_arr.append(val_running_loss)
        val_acc_arr.append(val_acc)

        print("epoch:", epoch+1, "training loss:", round(running_loss, 3), 'validation loss:', round(val_running_loss, 3), 'validation accuracy:', round(val_acc*100, 2))
    return train_loss_arr, val_loss_arr, val_acc_arr, model

def split_and_tokenise_glue(tokenizer, max_len,datasets):
    def get_tokenizer_fn(task):
        """Return a field‑aware tokenisation lambda for the task."""
        if task in ["sst2", "cola"]:
            return lambda batch: tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=max_len, return_token_type_ids=False)
        elif task in ["mrpc", "rte"]:
            return lambda batch: tokenizer(batch["sentence1"], batch["sentence2"], truncation=True, padding="max_length", max_length=max_len)
        elif task == "qqp":
            return lambda batch: tokenizer(batch["question1"], batch["question2"], truncation=True, padding="max_length", max_length=max_len)
        else:
            raise ValueError(f"Unsupported GLUE task: {task}")
    tokenized = {}
    for task in datasets:
        combined = load_dataset("glue", task, split="train+validation")
        combined = combined.shuffle(seed=SEED)

        # 1\) carve out held\u2011out test
        split1 = combined.train_test_split(test_size=SPLIT_TEST, seed=SEED)
        train_val = split1["train"]
        test_ds = split1["test"]

        # 2\) carve validation out of the remaining data
        val_ratio_relative = SPLIT_VAL / (1.0 - SPLIT_TEST)
        split2 = train_val.train_test_split(test_size=val_ratio_relative, seed=SEED)

        ds = DatasetDict({
            "train": split2["train"],
            "validation": split2["test"],
            "test": test_ds,
        })

        tok_fn = get_tokenizer_fn(task)
        ds = ds.map(tok_fn, batched=True, num_proc=NUM_PROC,remove_columns=[c for c in ds["train"].column_names if c not in ["label"]])
        ds.set_format(type="torch")
        tokenized[task] = ds
    return tokenized


def main():
    max_len = 128
    batch_size = 64
    epochs = 3
    lr = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["sst2","mrpc","qqp","rte","cola"]
    results = {task:{} for task in datasets}

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_ds = split_and_tokenise_glue(tokenizer,max_len,datasets)
    for task in datasets:
        ds = tokenized_ds[task]
        train_tensor_ds = ds["train"]
        val_tensor_ds = ds["validation"]
        test_tensor_ds = ds["test"]

        train_ds = CustomClassDataset(
            train_tensor_ds["input_ids"],
            train_tensor_ds["attention_mask"],
            train_tensor_ds["label"],
        )
        val_ds = CustomClassDataset(
            val_tensor_ds["input_ids"],
            val_tensor_ds["attention_mask"],
            val_tensor_ds["label"],
        )
        test_ds =  CustomClassDataset(
            test_tensor_ds["input_ids"],
            test_tensor_ds["attention_mask"],
            test_tensor_ds["label"],
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds,batch_size)

        encoder = DistilBertForMaskedLM(vocab_size=tokenizer.vocab_size, max_seq_length=max_len)
        state = torch.load("checkpoints/distilbert_pretrain.pt", map_location=device)
        encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith("mlm_head")}, strict=False)

        model = DistilBertForSequenceClassification(encoder, num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        train_loss_arr, val_loss_arr, val_acc_arr, model= train(model, train_loader, val_loader, epochs, optimizer, device)

        print("train_loss_arr:",train_loss_arr)
        print("val_loss_arr:", val_loss_arr)
        print("val_acc_arr:", val_acc_arr)

        print(f"\n>>> Cross‑task evaluation for model finetuned on {task} <<<")
        for other in datasets:
            other_ds = tokenized_ds[other]["test"]
            other_test = CustomClassDataset(other_ds["input_ids"], other_ds["attention_mask"], other_ds["label"])
            other_loader = DataLoader(other_test, batch_size=batch_size)
            o_loss, o_acc = evaluate(model, other_loader, device)
            print(f"Test on {other.upper():>5}: loss={o_loss:.4f}, acc={o_acc:.4f}")
            results[task][other] = {"loss":o_loss,"accuracy":o_acc}

        save_dir = os.path.join("checkpoints",task)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir,"distilbert_finetune.pt"))
        print(f"Weights saved {save_dir}/distilbert_finetune.pt")

    metrics_path = os.path.join("checkpoints", "glue_cross_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"All cross‑eval metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
