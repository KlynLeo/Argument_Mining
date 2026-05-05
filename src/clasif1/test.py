import time
from pathlib import Path
from transformers import pipeline
 
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "ro_bert_argumentative"
 
print(f"Model path: {MODEL_PATH}")
print(f"Path există: {MODEL_PATH.exists()}")
print(f"Fișiere în folder:")
for f in MODEL_PATH.iterdir():
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}  ({size_mb:.1f} MB)")
 
print("\nIncarcam modelul...")
t0 = time.time()
clf = pipeline("text-classification", model=str(MODEL_PATH))
print(f"Model incarcat in {time.time() - t0:.1f}s")
 
print("\nPredictii:")
print(clf("Energia nucleară este sigură pentru mediu."))
print(clf("Am o floare frumoasă."))