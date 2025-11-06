import json
import os
from collections import Counter

# --- File paths ---
base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
lexicon_path = os.path.join(base, "morphalou_dict.json")
input_path = os.path.join(base, "train.fr.clean.sample")

# --- Load the lexicon ---
with open(lexicon_path, "r", encoding="utf-8") as f:
    morphalou = json.load(f)

print(f"✅ Loaded {len(morphalou):,} entries from Morphalou.")

# --- Initialize counters ---
total_tokens = 0
known_tokens = 0
unknown_tokens = 0
unknown_counts = Counter()

# --- Token normalization helper ---
def normalize_token(tok):
    return tok.lower().strip(".,;:!?\"()[]{}")

# --- Process the corpus ---
with open(input_path, "r", encoding="utf-8") as fin:
    for line in fin:
        toks = line.strip().split()
        for tok in toks:
            t = normalize_token(tok)
            if not t:
                continue
            total_tokens += 1
            if t in morphalou:
                known_tokens += 1
            else:
                unknown_tokens += 1
                unknown_counts[t] += 1

# --- Compute coverage ---
coverage = known_tokens / total_tokens * 100 if total_tokens else 0

print("\n📊 --- Morphalou Coverage Report ---")
print(f"Total tokens: {total_tokens:,}")
print(f"Known tokens: {known_tokens:,}")
print(f"Unknown tokens: {unknown_tokens:,}")
print(f"Coverage: {coverage:.2f}% of tokens recognized")

# --- Show top unknown tokens ---
print("\n❗ Top 30 unknown tokens:")
for tok, freq in unknown_counts.most_common(30):
    print(f"{tok:20s} {freq}")
