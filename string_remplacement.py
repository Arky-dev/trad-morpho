import os

# --- Path to your file ---
path = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet\train.en.clean.sample"
STRING = "';"
TO_STRING = "'"

# --- Read, replace, and overwrite ---
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace both versions (some corpora use &apos or &apos;)
text = text.replace(STRING, TO_STRING)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Replaced all occurrences of {STRING} in {os.path.basename(path)}")
