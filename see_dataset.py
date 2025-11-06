import torch
from pprint import pprint

base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
data = torch.load(f"{base}\\train_dataset.pt", map_location="cpu")

print("✅ Loaded dataset keys:", list(data.keys()))
print()

# ---- Basic info ----
surface = data["surface"]
en = data["en"]
align = data["align"]
morph = data["morph"]
vocabs = data["vocabs"]

print("Surface shape:", surface.shape)
print("English shape:", en.shape)
print("Alignment shape:", align.shape)
print()

# ---- Vocabulary sizes ----
print("📘 Vocab sizes:")
print(f"  Surface: {len(vocabs['surface'])}")
print(f"  English: {len(vocabs['en'])}")
for f, v in vocabs["morph"].items():
    print(f"  Morph-{f}: {len(v)}")
print()

# ---- Inspect one sample ----
sample_idx = 0  # change to look at another sentence
surface_seq = surface[sample_idx]
en_seq = en[sample_idx]
align_mat = align[sample_idx]
morph_seq = {f: morph[f][sample_idx] for f in morph}

def decode(seq, vocab):
    inv = {i: t for t, i in vocab.items()}
    toks = [inv.get(int(i), "<unk>") for i in seq if i != 0]
    return toks

# Decode the first sample
src_words = decode(surface_seq, vocabs["surface"])
tgt_words = decode(en_seq, vocabs["en"])
print(f"🗣️  French surface (decoded): {' '.join(src_words)}")
print(f"🎯 English target (decoded): {' '.join(tgt_words)}")
print()

# Decode morphology features
print("🔬 Morphological features (for the same sentence):")
for f, seq in morph_seq.items():
    feats = decode(seq, vocabs["morph"][f])
    print(f"  {f:8}: {' '.join(feats)}")
print()

# Check alignment visualization (small text grid)
print("🕸️  Alignment pairs (nonzero indices):")
src_len, tgt_len = align_mat.shape[1], align_mat.shape[0]
pairs = [(i, j) for j in range(tgt_len) for i in range(src_len) if align_mat[j, i] > 0.5]
for (i, j) in pairs:
    if i < len(src_words) and j < len(tgt_words):
        print(f"  {src_words[i]} ↔ {tgt_words[j]}")
print()

# Optional: visualize alignment matrix (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(align_mat.numpy(), cmap="Greys", interpolation="nearest")
    plt.xticks(range(len(src_words)), src_words, rotation=45, ha="right")
    plt.yticks(range(len(tgt_words)), tgt_words)
    plt.title(f"Alignment Matrix — Sample {sample_idx}")
    plt.xlabel("French (source)")
    plt.ylabel("English (target)")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Matplotlib not installed — skipping visualization.")
