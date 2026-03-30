"""
prepare_dataset.py
------------------
Selecteaza propozitii echilibrat din 3 fisiere UKP TSV,
le traduce in romana cu Google Translate si exporta un CSV
gata de verificat in Label Studio.

Utilizare:
    python prepare_dataset.py

Output:
    dataset_ro.csv  (in acelasi folder)
"""

import pandas as pd
import time
from deep_translator import GoogleTranslator

# ──────────────────────────────────────────────
# CONFIGURARE
# ──────────────────────────────────────────────
INPUT_FILES = [
    "nuclear_energy.tsv",
    "minimum_wage.tsv",
    "school_uniforms.tsv",
]

LABELS = ["Argument_for", "Argument_against", "NoArgument"]

# cate propozitii per topic / per label / per split
SPLIT_COUNTS = {
    "train": 58,
    "val":   13,
    "test":  13,
}
# total: 3 topics x 3 labels x 84 = 756 propozitii

OUTPUT_FILE = "dataset_ro.csv"

# ──────────────────────────────────────────────
# SELECTIE
# ──────────────────────────────────────────────
def select_sentences(files, labels, split_counts):
    frames = []
    for filepath in files:
        df = pd.read_csv(filepath, sep="\t", quoting=3, on_bad_lines="skip")
        for label in labels:
            for split, n in split_counts.items():
                subset = df[
                    (df["annotation"] == label) &
                    (df["set"] == split)
                ]
                if len(subset) < n:
                    print(f"  ATENTIE: {filepath} / {label} / {split} "
                          f"are doar {len(subset)} propozitii (cerute {n})")
                    n = len(subset)
                sampled = subset.sample(n=n, random_state=42)
                frames.append(sampled)

    result = pd.concat(frames).reset_index(drop=True)
    return result[["topic", "sentence", "annotation", "set"]]


# ──────────────────────────────────────────────
# TRADUCERE
# ──────────────────────────────────────────────
def translate_batch(sentences, pause=0.5):
    """
    Traduce o lista de propozitii EN -> RO.
    pause = secunde intre requesturi (evita rate limiting)
    """
    translator = GoogleTranslator(source="en", target="ro")
    translated = []

    for i, sentence in enumerate(sentences):
        try:
            ro = translator.translate(sentence)
            translated.append(ro)
        except Exception as e:
            print(f"  Eroare la propozitia {i}: {e}")
            translated.append("")  # lasam gol, verifici manual

        # progress
        if (i + 1) % 50 == 0:
            print(f"  Traduse: {i+1}/{len(sentences)}")

        time.sleep(pause)

    return translated


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== PASUL 1: Selectie propozitii ===")
    df = select_sentences(INPUT_FILES, LABELS, SPLIT_COUNTS)
    print(f"Total selectat: {len(df)}")
    print(df["annotation"].value_counts().to_string())
    print(df["set"].value_counts().to_string())

    print("\n=== PASUL 2: Traducere EN -> RO ===")
    print("(poate dura 5-10 minute pentru 756 propozitii)")
    sentences_en = df["sentence"].tolist()
    sentences_ro = translate_batch(sentences_en, pause=0.3)

    print("\n=== PASUL 3: Asamblare CSV final ===")
    df["sentence_en"] = sentences_en
    df["sentence_ro"] = sentences_ro

    # reordonam coloanele
    df_final = df[["topic", "sentence_en", "sentence_ro", "annotation", "set"]]

    # salvam
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Salvat: {OUTPUT_FILE} ({len(df_final)} randuri)")

    # preview
    print("\nPrimele 3 randuri:")
    for _, row in df_final.head(3).iterrows():
        print(f"  [{row['annotation']}] EN: {row['sentence_en'][:60]}...")
        print(f"           RO: {row['sentence_ro'][:60]}...")
        print()

    print("Gata! Importa dataset_ro.csv in Label Studio pentru verificare.")