from __future__ import annotations
import json

import config
from pathlib import Path


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def cm_to_md(cm):
    return (
        "|  | pred 0 (non-arg) | pred 1 (arg) |\n"
        "|---|---|---|\n"
        f"| **true 0 (non-arg)** | {cm[0][0]} | {cm[0][1]} |\n"
        f"| **true 1 (arg)** | {cm[1][0]} | {cm[1][1]} |"
    )


def recall_per_class(cm):
    recall_for = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) else 0.0
    recall_against = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) else 0.0
    return recall_for, recall_against


def main():
    def resolve(preferred: Path, fallback_name: str) -> Path:
        if preferred and Path(preferred).exists():
            return Path(preferred)
        alt = config.RESULTS_DIR / fallback_name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Neither {preferred} nor {alt} found")

    zs_path = resolve(config.ZERO_SHOT_METRICS_PATH, "zero_shot_test.json")
    ft_val_path = resolve(config.FINETUNED_VAL_METRICS_PATH, "finetuned_val.json")
    ft_test_path = resolve(config.FINETUNED_TEST_METRICS_PATH, "finetuned_test.json")

    zs = load_metrics(zs_path)
    ft_val = load_metrics(ft_val_path)
    ft_test = load_metrics(ft_test_path)

    d_acc = ft_test["accuracy"] - zs["accuracy"]
    d_f1pos = ft_test["f1_pos"] - zs["f1_pos"]
    d_f1mac = ft_test["f1_macro"] - zs["f1_macro"]

    rec_for_zs, rec_against_zs = recall_per_class(zs["confusion_matrix"])
    rec_for_ft, rec_against_ft = recall_per_class(ft_test["confusion_matrix"])

    md = f"""# Comparație: Zero-shot vs Fine-tuned (clasificare argumentativ / non-argumentativ)

## Sinteza pe setul de test

| Metrica | Zero-shot (mDeBERTa) | Fine-tuned (RO-BERT) | Diferenta |
|---|---|---|---|
| n_samples | {zs['n_samples']} | {ft_test['n_samples']} | - |
| accuracy | {fmt(zs['accuracy'])} | **{fmt(ft_test['accuracy'])}** | {d_acc:+.4f} |
| f1_pos | {fmt(zs['f1_pos'])} | **{fmt(ft_test['f1_pos'])}** | {d_f1pos:+.4f} |
| **f1_macro** | {fmt(zs['f1_macro'])} | **{fmt(ft_test['f1_macro'])}** | **{d_f1mac:+.4f}** |
| precision_pos | {fmt(zs['precision_pos'])} | {fmt(ft_test['precision_pos'])} | {ft_test['precision_pos']-zs['precision_pos']:+.4f} |
| recall_pos | {fmt(zs['recall_pos'])} | {fmt(ft_test['recall_pos'])} | {ft_test['recall_pos']-zs['recall_pos']:+.4f} |
| recall_nonarg | {fmt(rec_for_zs)} | {fmt(rec_for_ft)} | {rec_for_ft-rec_for_zs:+.4f} |
| recall_arg | {fmt(rec_against_zs)} | {fmt(rec_against_ft)} | {rec_against_ft-rec_against_zs:+.4f} |

Fine-tuning-ul modifica performanta macro (f1_macro) cu **{d_f1mac*100:+.1f} puncte procentuale** pe test.

## Validare onesta: Val vs Test (fine-tuned)

| Metrica | Val | Test | Diferenta |
|---|---|---|---|
| n_samples | {ft_val['n_samples']} | {ft_test['n_samples']} | - |
| accuracy | {fmt(ft_val['accuracy'])} | {fmt(ft_test['accuracy'])} | {ft_test['accuracy']-ft_val['accuracy']:+.4f} |
| f1_pos (against) | {fmt(ft_val['f1_pos'])} | {fmt(ft_test['f1_pos'])} | {ft_test['f1_pos']-ft_val['f1_pos']:+.4f} |
| f1_macro | {fmt(ft_val['f1_macro'])} | {fmt(ft_test['f1_macro'])} | {ft_test['f1_macro']-ft_val['f1_macro']:+.4f} |

## Confusion matrices pe test

### Zero-shot (mDeBERTa)

{cm_to_md(zs['confusion_matrix'])}

### Fine-tuned (RO-BERT, 4 epochs)

{cm_to_md(ft_test['confusion_matrix'])}

## Classification reports detaliate (test)

### Zero-shot
```
{zs['classification_report']}
```

### Fine-tuned
```
{ft_test['classification_report']}
```

## Configurare experimentală

- Date: propoziții din `dataset_final.csv` (split-uri în `results`)
- Splituri și dimensiuni: folosiți fișierele din `results/` pentru valori exacte
- Clase: non-argumentativ (0) vs argumentativ (1)
- Model zero-shot: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- Model fine-tuned: vezi `config.RO_BERT_MODEL`
- Hiperparametri: lr=2e-5, batch=16, 4 epochs, weight_decay=0.01, warmup_ratio=0.1
- Selectie model: checkpoint-ul cu cel mai bun `f1_macro` pe val

## Concluzie

Zero-shot furnizeaza un baseline pentru polaritatea argumentului, iar fine-tuning-ul pe datele etichetate pentru/impotriva permite comparatia directa pe aceleasi metrici si aceeasi schema de etichete.
"""

    try:
        output_path = Path(config.COMPARISON_PATH)
    except Exception:
        output_path = config.RESULTS_DIR / "comparison.md"
    if not output_path.parent.exists():
        output_path = config.RESULTS_DIR / "comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Salvat raportul la: {output_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
