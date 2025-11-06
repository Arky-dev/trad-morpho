from simalign import SentenceAligner
import os

# Paths
base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
fr_file = os.path.join(base, "train.fr.morph.sample")
en_file = os.path.join(base, "train.en.clean.sample")
align_file = os.path.join(base, "train.align.sample")

# Initialize aligner
aligner = SentenceAligner(model="bert",
                          token_type="bpe",  # tokenization type
                          matching_methods="mai")

with open(fr_file, 'r', encoding='utf-8') as f_fr, \
     open(en_file, 'r', encoding='utf-8') as f_en, \
     open(align_file, 'w', encoding='utf-8') as fout:

    for idx, (fr_line, en_line) in enumerate(zip(f_fr, f_en)):
        fr_tokens = fr_line.strip().split()
        en_tokens = en_line.strip().split()

        # Compute alignment
        alignments = aligner.get_word_aligns(fr_tokens, en_tokens)

        # Format: i-j pairs
        pairs = []
        for i, j_list in enumerate(alignments['mwmf']):
            for j in j_list:
                pairs.append(f"{i}-{j}")
        fout.write(" ".join(pairs) + "\n")

