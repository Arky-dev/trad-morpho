import json
import os

# --- Paths ---
base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
lexicon_path = os.path.join(base, "morphalou_dict.json")
input_path = os.path.join(base, "test.fr.clean")
output_path = os.path.join(base, "test.fr.morph")

# --- Load the Morphalou lexicon ---
with open(lexicon_path, "r", encoding="utf-8") as f:
    morphalou = json.load(f)

print(f"✅ Loaded {len(morphalou)} entries from Morphalou lexicon.")

# --- Token annotation function ---
def annotate_token(tok: str) -> str:
    t = tok.lower().strip(".,;:!?\"()[]{}")

    # handle punctuation-only tokens or empty strings
    if not t:
        return tok

    if t in morphalou:
        d = morphalou[t]
        lemma = d.get("lemma", "")
        pos = d.get("pos", "")
        morph = d.get("morph", "")
        # Combine in a compact readable form
        return f"{lemma}_{pos}_{morph}".strip("_")
    else:
        return tok  # leave unchanged if unknown


# --- Annotate the corpus line by line ---
known = 0
unknown = 0
n_lines = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        toks = line.strip().split()
        annotated = []
        for t in toks:
            annotated_tok = annotate_token(t)
            if annotated_tok != t:
                known += 1
            else:
                unknown += 1
            annotated.append(annotated_tok)
        fout.write(" ".join(annotated) + "\n")
        n_lines += 1

# --- Summary ---
print(f"✅ Annotated {n_lines:,} lines.")
print(f"   → Known tokens: {known:,}")
print(f"   → Unknown tokens: {unknown:,}")
print(f"   → Output written to: {output_path}")
