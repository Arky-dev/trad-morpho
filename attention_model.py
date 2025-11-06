import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_surface, n_lemma, n_pos, emb_dim, hidden_dim):
        super().__init__()
        self.surface_embed = nn.Embedding(n_surface, emb_dim)
        self.lemma_embed = nn.Embedding(n_lemma, emb_dim)
        self.pos_embed = nn.Embedding(n_pos, emb_dim)
        self.merge = nn.Linear(emb_dim * 3, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, surface, lemma, pos):
        x = torch.cat([
            self.surface_embed(surface),
            self.lemma_embed(lemma),
            self.pos_embed(pos)
        ], dim=-1)
        x = torch.tanh(self.merge(x))
        outputs, hidden = self.rnn(x)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, gold_alignment=None):
        # hidden: (1, batch, hidden)
        hidden = hidden.permute(1, 0, 2)  # (batch, 1, hidden)
        score = self.v(torch.tanh(
            self.W1(encoder_outputs) + self.W2(hidden)
        )).squeeze(-1)
        attn = F.softmax(score, dim=-1)

        # Optional: alignment supervision
        attn_loss = None
        if gold_alignment is not None:
            gold = gold_alignment / (gold_alignment.sum(dim=1, keepdim=True) + 1e-8)
            attn_loss = F.kl_div(attn.log(), gold, reduction='batchmean')
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn, attn_loss

class Decoder(nn.Module):
    def __init__(self, n_tgt, emb_dim, hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(n_tgt, emb_dim)
        self.attention = attention
        self.rnn = nn.GRU(hidden_dim + emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, n_tgt)

    def forward(self, input_tok, hidden, encoder_outputs, gold_alignment=None):
        emb = self.embedding(input_tok).unsqueeze(1)
        context, attn, attn_loss = self.attention(hidden, encoder_outputs, gold_alignment)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        pred = self.fc_out(torch.cat([output.squeeze(1), context], dim=1))
        return pred, hidden, attn, attn_loss
