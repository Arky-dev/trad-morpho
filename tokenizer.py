import unicodedata
from sacremoses import MosesTokenizer

# Paths
base_path = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
fr_in = f"{base_path}\\test.fr"
en_in = f"{base_path}\\test.en"
fr_out = f"{base_path}\\test.fr.clean"
en_out = f"{base_path}\\test.en.clean"

# Initialize tokenizers
tok_fr = MosesTokenizer(lang='fr')
tok_en = MosesTokenizer(lang='en')

def normalize(text: str) -> str:
    text = text.strip()
    # NFKC normalization (consistent accents, punctuation, etc.)
    text = unicodedata.normalize('NFKC', text)
    # Remove weird control chars
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in ('\t', '\n'))
    return text.lower().strip(".,;:!?\"'()[]{}")

with open(fr_in, 'r', encoding='utf-8') as f_fr, open(en_in, 'r', encoding='utf-8') as f_en, open(fr_out, 'w', encoding='utf-8') as o_fr, open(en_out, 'w', encoding='utf-8') as o_en:

    n_bad = 0
    n_total = 0

    for line_fr, line_en in zip(f_fr, f_en):
        n_total += 1
        line_fr = normalize(line_fr)
        line_en = normalize(line_en)

        # Skip empty or extremely short/long lines
        if not line_fr or not line_en:
            n_bad += 1
            continue
        if len(line_fr.split()) > 200 or len(line_en.split()) > 200:
            n_bad += 1
            continue
        if len(line_fr.split()) / max(1, len(line_en.split())) > 3.0 or \
           len(line_en.split()) / max(1, len(line_fr.split())) > 3.0:
            n_bad += 1
            continue

        # Tokenize
        line_fr_tok = tok_fr.tokenize(line_fr, return_str=True)
        line_en_tok = tok_en.tokenize(line_en, return_str=True)

        o_fr.write(line_fr_tok + '\n')
        o_en.write(line_en_tok + '\n')

print(f"Done! Total {n_total} pairs processed, {n_bad} filtered.")
print(f"Clean files written to:\n  {fr_out}\n  {en_out}")
