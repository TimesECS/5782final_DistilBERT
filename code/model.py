import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(max_seq_length, 1)
        div_term = torch.exp( 
            -1 * (torch.arange(0, d_model, 2).float()/d_model) * math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds the positional encoding to the model input x.
        """
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Define layers W_q, W_k, W_v, and W_o
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V, mask=None):
        """
        Returns single-headed attention between Q, K, and V.
        """
        # Compute attention
        qk = torch.matmul(Q, K.transpose(-2, -1))
        dk = torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            qk = qk.masked_fill(mask == 0, float("-inf"))
        softmax = F.softmax(qk/torch.sqrt(dk), dim=-1)
        attention = torch.matmul(softmax, V)
        return attention
    
    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x, mask=None):
        # Implement forward pass
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        x = self.compute_attention(Q, K, V, mask)
        x = self.combine_heads(x)
        x = self.W_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        # Define the network
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Implement feed forward pass
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        # Define the encoder layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask=None):
        ## Implement the forward function
        out = self.self_attn(x, mask)
        out = self.dropout(out)
        x = self.norm1(x + out)
        out = self.feed_forward(x)
        out = self.dropout(out)
        x = self.norm2(x + out) 
        return x


def compute_loss(student_logits, mlm_labels, *, student_hiddens=None,
                teacher_logits=None, teacher_hiddens=None, 
                temperature=2.0, alpha=5.0, beta=2.0, gamma=1.0):
    # L_mlm
    ce_fct = nn.CrossEntropyLoss(ignore_index=-100)
    L_mlm = ce_fct(student_logits.view(-1, student_logits.size(-1)), mlm_labels.view(-1))

    if teacher_logits is None or teacher_hiddens is None:
        return {
            "L_distill": torch.tensor(0.0, device=student_logits.device),
            "L_mlm": L_mlm,
            "L_cos": torch.tensor(0.0, device=student_logits.device),
            "loss": L_mlm
        }

    # L_distill
    kl_fct = nn.KLDivLoss(reduction="batchmean")
    student_log_prob = F.log_softmax(student_logits/temperature, dim=-1)
    with torch.no_grad():
        teacher_prob = F.softmax(teacher_logits/temperature, dim=-1)
    L_distill = kl_fct(student_log_prob, teacher_prob) * (temperature**2)

    # L_cos
    step = max(len(teacher_hiddens)//len(student_hiddens), 1)
    teacher_aligned = [teacher_hiddens[i] for i in range(0, step * len(student_hiddens), step)]

    cos_fct = nn.CosineEmbeddingLoss()
    num_layers = len(student_hiddens)
    L_cos = 0.0
    for s, t in zip(student_hiddens, teacher_aligned):
        L_cos += cos_fct(
            s.view(-1, s.size(-1)),
            t.detach().view(-1, t.size(-1)),
            torch.ones(s.size(0) * s.size(1), device=s.device),
        )
    L_cos /= num_layers

    L_triple = alpha * L_distill + beta * L_mlm + gamma * L_cos
    return {
        "L_distill": L_distill,
        "L_mlm": L_mlm,
        "L_cos": L_cos,
        "loss": L_triple
    }


class DistilBertForMaskedLM(nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model=768, num_heads=12, num_layers=6, d_ff=3072, p=0.1):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, p) for _ in range(num_layers)]
        )

        self.mlm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.mlm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids, attention_mask, labels=None, *, teacher_logits=None, teacher_hiddens=None, return_hidden=False,
                temperature=2.0, alpha=5.0, beta=2.0, gamma=1.0):
        hiddens = []
        x = self.token_embeddings(input_ids)
        x = self.position_embeddings(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=attention_mask)
            hiddens.append(x)

        logits = self.mlm_head(x)

        outputs = {"logits": logits}
        if return_hidden:
            outputs["hiddens"] = hiddens
        
        if labels is not None:
            losses = compute_loss(
                logits,
                labels,
                student_hiddens=hiddens,
                teacher_logits=teacher_logits,
                teacher_hiddens=teacher_hiddens,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            outputs.update(losses)

        return outputs


class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.cls_head = nn.Linear(encoder.mlm_head.in_features, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        last_hidden = self.encoder(
            input_ids,
            attention_mask,
            return_hidden = True
        )["hiddens"][-1]

        logits = self.cls_head(last_hidden[:, 0])
        outputs = {"logits": logits}

        if labels is not None:
            ce_fct = nn.CrossEntropyLoss()
            loss = ce_fct(logits, labels)
            outputs.update({"loss": loss, "preds": torch.argmax(logits, 1)})

        return outputs
