import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm.auto import tqdm
from model import DistilBertForMaskedLM

WIKI_DUMP_ID = "20220301.en"
SEED = 1
SPLIT_VAL = 0.1
SPLIT_TEST = 0.1
NUM_PROC = 16
TEACHER_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128

class CustomMLMDataset(Dataset):
    def __init__(self, ids, attention_masks, tokenizer, max_len, mlm_prob=0.15):
        self.ids = ids
        self.attention_masks = attention_masks
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.ids)

    def _mask(self, ids):
        labels = ids.clone()
        prob = torch.full(labels.shape, self.mlm_prob)
        prob[self.tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True)] = 0.0
        mask = torch.bernoulli(prob).bool()
        labels[~mask] = -100

        ids[mask] = self.tokenizer.mask_token_id
        rand = torch.bernoulli(torch.full(ids.shape, 0.5)).bool() & mask
        random_tokens = torch.randint(self.vocab_size, ids.shape, dtype=torch.long)
        ids[rand] = random_tokens[rand]
        return ids, labels

    def __getitem__(self, idx):
        source_ids = torch.tensor(self.ids[idx])
        source_mask = torch.tensor(self.attention_masks[idx])
        masked_ids, labels = self._mask(source_ids.clone())
        return {
            "source_ids": masked_ids,
            "source_mask": source_mask,
            "label": labels,
        }


def evaluate(student, val_loader, device):
    student.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            out = student(
                batch["source_ids"].to(device),
                batch["source_mask"].to(device),
                batch["label"].to(device),
            )
            val_running_loss += out["loss"].item()
    return val_running_loss/len(val_loader)


def train(student, teacher, train_loader, val_loader, epochs, optimizer, device, *, temperature=2.0, alpha=5.0, beta=2.0, gamma=1.0):
    train_loss_arr = []
    val_loss_arr = []
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch["source_ids"].to(device),
                    attention_mask=batch["source_mask"].to(device),
                )
                t_logits = t_out.logits
                t_hiddens = t_out.hidden_states

            s_out = student(
                batch["source_ids"].to(device),
                batch["source_mask"].to(device),
                labels=batch["label"].to(device),
                teacher_logits=t_logits,
                teacher_hiddens=t_hiddens,
                return_hidden=True,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            loss = s_out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_running_loss = evaluate(student, val_loader, device)
        running_loss /= len(train_loader)
        train_loss_arr.append(running_loss)
        val_loss_arr.append(val_running_loss)

        print("epoch:", epoch+1, "training loss:", round(running_loss, 3), 'validation loss:', round(val_running_loss, 3))

    return train_loss_arr, val_loss_arr

def load_wiki_from_remote():
    print("\nLoading English Wikipedia dump …")
    wiki = load_dataset("wikipedia", WIKI_DUMP_ID, split="train",trust_remote_code=True)
    wiki = wiki.shuffle(seed=SEED)

    return wiki

def test(student, test_loader, device):
    """Evaluate *once* on the held-out test split."""
    student.eval()
    running = 0.0
    with torch.no_grad():
        for batch in test_loader:
            out = student(
                batch["source_ids"].to(device),
                batch["source_mask"].to(device),
                batch["label"].to(device),
            )
            running += out["loss"].item()
    return running / len(test_loader)

def split_and_tokenize_dataset(wiki, tokenizer, max_len):


    split1 = wiki.train_test_split(test_size=SPLIT_TEST, seed=SEED)
    train_val = split1["train"]
    test_ds = split1["test"]

    val_ratio_relative = SPLIT_VAL / (1.0 - SPLIT_TEST)
    split2 = train_val.train_test_split(test_size=val_ratio_relative, seed=SEED)
    train_ds = split2["train"]
    val_ds = split2["test"]

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_token_type_ids=False
        )

    ds = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })
    ds = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=NUM_PROC)
    ds.set_format(type="torch")
    return ds

def main():
    max_len = MAX_LEN
    batch_size = 32
    epochs = 1
    lr = 5e-5
    temperature = 2.0
    alpha = 5.0
    beta = 2.0
    gamma = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_path = Path("./tokenized")
    tokenizer = BertTokenizerFast.from_pretrained(TEACHER_MODEL_NAME)

    if not tokenized_path.exists():
        wiki = load_dataset("wikipedia",WIKI_DUMP_ID,split="train",trust_remote_code=True)
        ds = split_and_tokenize_dataset(wiki,tokenizer,max_len)
        os.makedirs(tokenized_path, exist_ok=True)
        ds.save_to_disk(tokenized_path)


    ds = DatasetDict.load_from_disk(str(tokenized_path))
    train_tensor_ds = ds["train"]
    val_tensor_ds = ds["validation"]
    test_tensor_ds = ds["test"]

    train_ds = CustomMLMDataset(train_tensor_ds["input_ids"], train_tensor_ds["attention_mask"], tokenizer,max_len=max_len)
    val_ds = CustomMLMDataset(val_tensor_ds["input_ids"], val_tensor_ds["attention_mask"], tokenizer,max_len)
    test_ds = CustomMLMDataset(test_tensor_ds["input_ids"], test_tensor_ds["attention_mask"], tokenizer,max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds,batch_size)

    teacher = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).eval().to(device)
    # student = DistilBertForMaskedLM(vocab_size=tokenizer.vocab_size, max_seq_length=max_len).to(device)
    student = DistilBertForMaskedLM(vocab_size=30522, max_seq_length=max_len).to(device)
    optimizer = opt.AdamW(student.parameters(), lr=lr)

    train_loss_arr, val_loss_arr = train(student, teacher, train_loader, val_loader, epochs, optimizer, device,
                                         temperature=temperature, alpha=alpha, beta=beta, gamma=gamma)
    
    test_loss = test(student, test_loader, device)
    print(f"► Test loss: {test_loss:.3f}") 
    print("train_loss_arr:", train_loss_arr)
    print("val_loss_arr:", val_loss_arr)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), "checkpoints/distilbert_pretrain.pt")
    print("Weights saved to checkpoints/distilbert_pretrain.pt")


if __name__ == "__main__":
    main()
