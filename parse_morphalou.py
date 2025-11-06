import csv
import json
import os

# --- File paths ---
base = r"C:\Users\paula\Documents\[1] School\[0] X\[2] 2A\HSS\projet"
csv_path = os.path.join(base, "Morphalou3.1_CSV.csv")  # update name if needed
out_path = os.path.join(base, "morphalou_dict.json")

# --- Step 1: Find the start of data section ---
with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Find where "LEMME" header appears
start_index = None
for i, line in enumerate(lines):
    if line.strip().startswith("LEMME"):
        start_index = i + 1
        break

if start_index is None:
    raise ValueError("Could not find 'LEMME' header in file.")

data_lines = lines[start_index:]

# --- Step 2: Parse using csv.reader ---
reader = csv.reader(data_lines, delimiter=';')

# Skip empty lines or comment lines
rows = [row for row in reader if any(cell.strip() for cell in row)]

# The first row after "LEMME;;;;;;;;;FLEXION;;;;;;;" describes the columns
header = [
    "LEMME_GRAPHIE", "LEMME_ID", "CATÉGORIE", "SOUS_CATÉGORIE", "LOCUTION",
    "GENRE_LEMME", "AUTRES_LEMMES", "PHON_LEMME", "ORIGINES_LEMME",
    "FLEXION_GRAPHIE", "FLEXION_ID", "NOMBRE", "MODE", "GENRE_FLEXION",
    "TEMPS", "PERSONNE", "PHON_FLEXION", "ORIGINES_FLEXION"
]

# --- Step 3: Build dictionary ---
entries = {}
current_lemma = None
current_pos = None
current_gender = None

for row in rows:
    if len(row) < 10:
        continue  # skip incomplete rows

    # Left side (lemma-level info)
    lemma_graphie = row[0].strip().lower()
    categorie = row[2].strip().capitalize() or current_pos
    genre_lemme = row[5].strip().lower() or current_gender

    # Update current lemma context when a new lemma line starts
    if lemma_graphie:
        current_lemma = lemma_graphie
        current_pos = categorie
        current_gender = genre_lemme

    # Right side (inflected form info)
    if len(row) >= 10:
        flex_form = row[9].strip().lower()
        if not flex_form:
            continue  # no inflected form

        number = row[11].strip().lower()
        mode = row[12].strip().lower()
        gender_flex = row[13].strip().lower()
        tense = row[14].strip().lower()
        person = row[15].strip().lower()

        # Combine features into a morphological string
        feats = "_".join(filter(None, [mode, tense, number, person, gender_flex]))

        entries[flex_form] = {
            "lemma": current_lemma,
            "pos": current_pos,
            "morph": feats
        }

print(f"✅ Loaded {len(entries)} inflected forms from Morphalou3.1")

# --- Step 4: Save dictionary ---
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"✅ Dictionary saved to {out_path}")
