from __future__ import annotations

import time
from pathlib import Path
from transformers import pipeline

import config


from pathlib import Path
import time
from transformers import pipeline

import config


MODEL_PATH = Path(config.FINETUNED_MODEL_DIR)

print(f"Model path: {MODEL_PATH}")
print(f"Path există: {MODEL_PATH.exists()}")
print("Fișiere în folder:")
if MODEL_PATH.exists():
    for f in MODEL_PATH.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")

print("\nÎncarc modelul...")
t0 = time.time()
clf = pipeline("text-classification", model=str(MODEL_PATH))
print(f"Model încărcat în {time.time() - t0:.1f}s")

print("\nPredicții exemplu:")
examples = [
    "Energia nucleară este sigură pentru mediu.",
    "Programele de creștere a salariului minim sunt dăunătoare economiei.",
]
for ex in examples:
    print(ex)
    print(clf(ex))
    print(f"Model loaded in {time.time()-t0:.1f}s")
