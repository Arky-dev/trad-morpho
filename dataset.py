import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"

paths = {
    "fr_surface": f"{base}\\train.fr.clean.sample",
    "fr_morph": f"{base}\\train.fr.morph.sample",
    "en": f"{base}\\train.en.clean.sample",
    "align": f"{base}\\train.align.sample",
}

# ---------- Load data ----------
with open(paths["fr_surface"], encoding="utf-8") as f:
    fr_surface = [l.strip().split() for l in f]

with open(paths["fr_morph"], encoding="utf-8") as f:
    fr_morph = [l.strip().split() for l in f]

with open(paths["en"], encoding="utf-8") as f:
    en_sents = [l.strip().split() for l in f]

with open(paths["align"], encoding="utf-8") as f:
    aligns = [l.strip() for l in f]

assert len(fr_surface) == len(fr_morph) == len(en_sents) == len(aligns)
print(f"✅ Loaded {len(fr_surface)} sentence pairs.")


# ---------- Parse morphological tokens ----------
def parse_morph_token(tok):
    parts = tok.split("_")
    while len(parts) < 7:
        parts.append("_")
    lemma, pos, mood, tense, number, person, gender = parts[:7]
    return {
        "lemma": lemma,
        "pos": pos,
        "mood": mood,
        "tense": tense,
        "number": number,
        "person": person,
        "gender": gender
    }

# Extract fields per sentence
fields = ["lemma", "pos", "mood", "tense", "number", "person", "gender"]
morph_features = {f: [] for f in fields}

for sent in fr_morph:
    parsed = [parse_morph_token(tok) for tok in sent]
    for f in fields:
        morph_features[f].append([p[f] for p in parsed])


# ---------- Build vocabularies ----------
def build_vocab(seqs, min_freq=1):
    counter = Counter(tok for sent in seqs for tok in sent)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for tok, c in counter.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

vocab_surface = build_vocab(fr_surface)
vocab_en = build_vocab(en_sents)
vocab_morph = {f: build_vocab(morph_features[f]) for f in fields}


# ---------- Encode sequences ----------
def encode(sent, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in sent]

def add_specials(seq):
    return [1] + seq + [2]  # <sos> ... <eos>

encoded_surface = [torch.tensor(add_specials(encode(s, vocab_surface))) for s in fr_surface]
encoded_en = [torch.tensor(add_specials(encode(s, vocab_en))) for s in en_sents]

encoded_morph = {
    f: [torch.tensor(add_specials(encode(s, vocab_morph[f]))) for s in morph_features[f]]
    for f in fields
}


# ---------- Parse alignments ----------
def parse_align(line, src_len, tgt_len):
    mat = torch.zeros(tgt_len, src_len)
    for pair in line.split():
        try:
            s, t = pair.split("-")
            s, t = int(s), int(t)
            if t < tgt_len and s < src_len:
                mat[t, s] = 1
        except ValueError:
            continue
    return mat

align_mats = []
for i, line in enumerate(aligns):
    src_len = len(fr_surface[i])
    tgt_len = len(en_sents[i])
    align_mats.append(parse_align(line, src_len + 2, tgt_len + 2))


# ---------- Pad all tensors ----------
pad = lambda seqs: pad_sequence(seqs, batch_first=True, padding_value=0)
surface_batch = pad(encoded_surface)
en_batch = pad(encoded_en)

morph_batches = {f: pad(encoded_morph[f]) for f in fields}

# Pad alignments manually
max_tgt, max_src = en_batch.shape[1], surface_batch.shape[1]
align_batch = torch.zeros(len(align_mats), max_tgt, max_src)
for i, mat in enumerate(align_mats):
    t, s = mat.shape
    align_batch[i, :t, :s] = mat

print("✅ Encoded dataset")
print(f"Surface: {surface_batch.shape}, English: {en_batch.shape}")
print(f"Alignments: {align_batch.shape}")


# ---------- Save to disk ----------
torch.save({
    "surface": surface_batch,
    "morph": morph_batches,
    "en": en_batch,
    "align": align_batch,
    "vocabs": {
        "surface": vocab_surface,
        "en": vocab_en,
        "morph": vocab_morph
    }
}, f"{base}\\train_dataset.pt")

print("💾 Saved dataset to train_dataset.pt")