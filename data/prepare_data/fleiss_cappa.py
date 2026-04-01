import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

file_path = "adnotat_pt_fleiss.csv"

label_cols = [
    "label_calin",
    "label_diana",
    "label_mihai",
    "label_daniel",
    "label_tudor"
]

df = pd.read_csv(file_path)

df = df[(df[label_cols] != "").all(axis=1)]

for col in label_cols:
    df[col] = df[col].str.strip().str.lower()

labels = sorted(set(df[label_cols].values.flatten()))

print("Labels detectate:", labels)

matrix = []

for _, row in df.iterrows():
    counts = [sum(row[col] == label for col in label_cols) for label in labels]
    matrix.append(counts)

kappa = fleiss_kappa(matrix)

print("\nFleiss' kappa:", round(kappa, 4))