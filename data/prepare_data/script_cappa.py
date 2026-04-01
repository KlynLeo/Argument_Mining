import pandas as pd
import random

# === CONFIG ===
input_file = "dataset_ro_curatat.csv"
output_file = "cappa.csv"
samples_per_topic = 17
total_samples = 51
random_seed = 42

random.seed(random_seed)

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === SELECT DATA ===
selected_rows = []

for topic, group in df.groupby("topic"):
    # amestecăm datele
    group = group.sample(frac=1, random_state=random_seed)

    # încercăm să diversificăm label-urile
    balanced = []

    for label, subg in group.groupby("annotation"):
        balanced.extend(subg.head(6).to_dict("records"))  # max 6 per label

    # dacă nu avem destule, completăm
    if len(balanced) < samples_per_topic:
        remaining = group[~group.index.isin([r.name if hasattr(r, 'name') else None for r in balanced])]
        remaining = remaining.to_dict("records")
        balanced.extend(remaining[:samples_per_topic - len(balanced)])

    selected_rows.extend(balanced[:samples_per_topic])

# dacă sunt mai mult de 51, tăiem
selected_rows = selected_rows[:total_samples]

new_df = pd.DataFrame(selected_rows)

# === KEEP ONLY REQUIRED COLUMNS ===
new_df = new_df[["topic", "sentence_ro", "annotation"]]

# === ADD EMPTY LABEL COLUMNS ===
new_df["label_calin"] = ""
new_df["label_diana"] = ""
new_df["label_mihai"] = ""
new_df["label_daniel"] = ""
new_df["label_tudor"] = ""

# === SAVE ===
new_df.to_csv(output_file, index=False)

print(f"Saved {len(new_df)} rows to {output_file}")