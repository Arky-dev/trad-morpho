# model_fixed.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

# ---------------- Dataset ----------------
class TranslationDataset(Dataset):
    def __init__(self, data):
        # data["surface"], data["en"], data["align"] are tensors
        self.surface = data["surface"]
        self.morph = data["morph"]
        self.en = data["en"]
        # ensure float alignments
        self.align = data["align"].float()

    def __len__(self):
        return self.surface.size(0)

    def __getitem__(self, idx):
        return (
            self.surface[idx],
            {f: self.morph[f][idx] for f in self.morph},
            self.en[idx],
            self.align[idx],
        )

# ---------------- Model ----------------
class MorphAwareTranslator(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embeddings
        self.surface_emb = nn.Embedding(vocab_sizes["surface"], embed_dim, padding_idx=0)
        self.morph_embs = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes["morph"][f], embed_dim // 4, padding_idx=0)
            for f in vocab_sizes["morph"]
        })
        self.en_emb = nn.Embedding(vocab_sizes["en"], embed_dim, padding_idx=0)

        # Encoder
        input_size = embed_dim + len(self.morph_embs) * (embed_dim // 4)
        self.encoder = nn.GRU(input_size, hidden_dim, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder
        # decoder input will be: [embedding(y_t) ; context (H)]
        self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_sizes["en"])

        # Attention for inference (additive)
        self.attn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x_surface, x_morph, y, align=None, use_gold_align=True):
        """
        x_surface: (B, S)
        x_morph: dict of (B, S) for each morph feature
        y: (B, T) full target sequence including <sos> and <eos>
        align: (B, T, S) alignment matrices (may include positions for <sos> and <eos>)
        """
        device = x_surface.device

        # --- Encoder ---
        surf = self.surface_emb(x_surface)  # (B, S, E)
        morph_vecs = [self.morph_embs[f](x_morph[f]) for f in x_morph]  # list of (B, S, Em)
        morph_cat = torch.cat(morph_vecs, dim=-1) if morph_vecs else torch.zeros_like(surf)
        enc_in = torch.cat([surf, morph_cat], dim=-1)  # (B, S, input_size)
        enc_out, _ = self.encoder(enc_in)  # (B, S, 2*H)
        enc_out = self.enc_proj(enc_out)   # (B, S, H)

        # --- Decoder (teacher forcing) ---
        B, T = y.size()
        h = enc_out.mean(dim=1).unsqueeze(0)  # (1, B, H) initial hidden
        dec_in = self.en_emb(y[:, :-1])       # (B, T-1, E)
        logits = []

        for t in range(T - 1):
            dec_t = dec_in[:, t].unsqueeze(1)  # (B,1,E)

            if use_gold_align and align is not None:
                # USE GOLD ALIGN for *the target position t+1* (because dec_t is y[:,t] -> predicts y[:,t+1])
                # align shape: (B, T, S). We want the row corresponding to the target token being predicted.
                a_t = align[:, t + 1, :]  # (B, S)
                # normalize each row to sum to 1 (to get convex combination)
                denom = a_t.sum(-1, keepdim=True)
                a_t = a_t / (denom + 1e-8)
                context = torch.bmm(a_t.unsqueeze(1), enc_out)  # (B,1,H)
            else:
                # Learned attention (inference / fallback)
                h_rep = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)  # (B, S, H)
                energy = torch.tanh(self.attn(torch.cat([h_rep, enc_out], dim=-1)))  # (B,S,H)
                scores = self.v(energy).squeeze(-1)  # (B, S)
                a_t = F.softmax(scores, dim=-1)
                context = torch.bmm(a_t.unsqueeze(1), enc_out)  # (B,1,H)

            dec_input = torch.cat([dec_t, context], dim=-1)  # (B,1, E+H)
            out, h = self.decoder(dec_input, h)  # out: (B,1,H)
            logits.append(self.fc_out(out))      # (B,1,V)

        logits = torch.cat(logits, dim=1)  # (B, T-1, V)
        return logits

    def encode_only(self, x_surface, x_morph):
        """Return encoder outputs and initial hidden state for inference"""
        surf = self.surface_emb(x_surface)
        morph_vecs = [self.morph_embs[f](x_morph[f]) for f in x_morph]
        morph_cat = torch.cat(morph_vecs, dim=-1) if morph_vecs else torch.zeros_like(surf)
        enc_in = torch.cat([surf, morph_cat], dim=-1)
        enc_out, _ = self.encoder(enc_in)
        enc_out = self.enc_proj(enc_out)
        h = enc_out.mean(dim=1).unsqueeze(0)
        return enc_out, h

# ---------------- Translate helper ----------------
def translate_from_surface(model, sentence_tokens_ids, vocab_en, device, max_len=40):
    """
    Greedy decoding given surface token ids (already include <sos> and <eos> if desired).
    sentence_tokens_ids : 1D list/torch tensor (S,)
    """
    model.eval()
    with torch.no_grad():
        # Build batch of size 1
        x_surface = torch.tensor([sentence_tokens_ids], device=device)
        # create dummy morph features (zeros) for each morph feature in model
        x_morph = {f: torch.zeros_like(x_surface) for f in model.morph_embs.keys()}

        enc_out, h = model.encode_only(x_surface, x_morph)  # enc_out: (1, S, H), h: (1,1,H)

        # start token
        sos = torch.tensor([[1]], device=device)
        dec_input = model.en_emb(sos)  # (1,1,E)
        outputs = []

        for _ in range(max_len):
            # compute learned attention (no gold align at inference)
            h_rep = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)  # (1,S,H)
            energy = torch.tanh(model.attn(torch.cat([h_rep, enc_out], dim=-1)))
            scores = model.v(energy).squeeze(-1)  # (1,S)
            a_t = F.softmax(scores, dim=-1)
            context = torch.bmm(a_t.unsqueeze(1), enc_out)  # (1,1,H)

            dec_input_full = torch.cat([dec_input, context], dim=-1)
            out, h = model.decoder(dec_input_full, h)
            logits = model.fc_out(out[:, -1])  # (1, V)
            next_tok = logits.argmax(-1)  # (1,)
            outputs.append(next_tok.item())
            if next_tok.item() == 2:
                break
            dec_input = model.en_emb(next_tok.unsqueeze(0))

    inv_vocab = {v: k for k, v in vocab_en.items()}
    return " ".join(inv_vocab.get(i, "<unk>") for i in outputs)


# ---------------- Training entrypoint ----------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.dataset, map_location=device)

    # compute vocab sizes
    vocab_sizes = {
        "surface": len(data["vocabs"]["surface"]),
        "en": len(data["vocabs"]["en"]),
        "morph": {f: len(v) for f, v in data["vocabs"]["morph"].items()}
    }

    dataset = TranslationDataset(data)
    # allow tiny overfit debugging: use small subset if requested
    # if args.overfit:
    #     n = min(4, len(dataset))
    #     indices = list(range(n))
    #     subset = torch.utils.data.Subset(dataset, indices)
    #     loader = DataLoader(subset, batch_size=2, shuffle=True)
    #     print(f"Overfitting on {n} examples (batch_size=2).")
    # else:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MorphAwareTranslator(vocab_sizes, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x_surface, x_morph, y, align in loader:
            x_surface = x_surface.to(device)
            y = y.to(device)
            align = align.to(device)
            x_morph = {k: v.to(device) for k, v in x_morph.items()}

            optimizer.zero_grad()
            logits = model(x_surface, x_morph, y, align, use_gold_align=True)
            # logits: (B, T-1, V), targets: y[:,1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} — loss: {avg:.4f}")

        # quick sanity print (translate first example)
        if args.overfit and epoch % 5 == 0:
            # translate the first example using greedy decode
            sample_surface = dataset[0][0]  # tensor
            # convert to python list and remove padding if any
            s_list = sample_surface.cpu().tolist()
            print(" Sample translation:", translate_from_surface(model, s_list, data["vocabs"]["en"], device))

    # save checkpoint (also save vocabs)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocabs": data["vocabs"],
        "vocab_sizes": vocab_sizes,
        "config": {"embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim}
    }, "checkpoints/model.pt")
    print("✅ Model saved to checkpoints/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="train_dataset.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--overfit", action="store_true", help="Overfit tiny subset for debugging")
    args = parser.parse_args()
    main(args)
