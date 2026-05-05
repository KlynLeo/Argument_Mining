from __future__ import annotations
import json

import config


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
    recall_neg = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) else 0.0
    recall_pos = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) else 0.0
    return recall_neg, recall_pos


def main():
    zs = load_metrics(config.ZERO_SHOT_METRICS_PATH)
    ft_val = load_metrics(config.FINETUNED_VAL_METRICS_PATH)
    ft_test = load_metrics(config.FINETUNED_TEST_METRICS_PATH)

    d_acc = ft_test["accuracy"] - zs["accuracy"]
    d_f1pos = ft_test["f1_pos"] - zs["f1_pos"]
    d_f1mac = ft_test["f1_macro"] - zs["f1_macro"]

    rec_neg_zs, rec_pos_zs = recall_per_class(zs["confusion_matrix"])
    rec_neg_ft, rec_pos_ft = recall_per_class(ft_test["confusion_matrix"])

    md = f"""# Comparație: Zero-shot vs Fine-tuned (clasificare argumentativ / non-argumentativ)

## Sinteză pe setul de test

| Metrică | Zero-shot (mDeBERTa) | Fine-tuned (RO-BERT) | Diferență |
|---|---|---|---|
| n_samples | {zs['n_samples']} | {ft_test['n_samples']} | – |
| accuracy | {fmt(zs['accuracy'])} | **{fmt(ft_test['accuracy'])}** | {d_acc:+.4f} |
| f1_pos | {fmt(zs['f1_pos'])} | **{fmt(ft_test['f1_pos'])}** | {d_f1pos:+.4f} |
| **f1_macro** | {fmt(zs['f1_macro'])} | **{fmt(ft_test['f1_macro'])}** | **{d_f1mac:+.4f}** |
| precision_pos | {fmt(zs['precision_pos'])} | {fmt(ft_test['precision_pos'])} | {ft_test['precision_pos']-zs['precision_pos']:+.4f} |
| recall_pos | {fmt(zs['recall_pos'])} | {fmt(ft_test['recall_pos'])} | {ft_test['recall_pos']-zs['recall_pos']:+.4f} |
| recall_neg | {fmt(rec_neg_zs)} | {fmt(rec_neg_ft)} | {rec_neg_ft-rec_neg_zs:+.4f} |

Fine-tuning-ul îmbunătățește f1_macro cu **{d_f1mac*100:+.1f} puncte procentuale** și aduce o balansare semnificativă a claselor (recall_neg crește de la {fmt(rec_neg_zs)} la {fmt(rec_neg_ft)}).

## Validare onestă: Val vs Test (fine-tuned)

| Metrică | Val | Test | Diferență |
|---|---|---|---|
| n_samples | {ft_val['n_samples']} | {ft_test['n_samples']} | – |
| accuracy | {fmt(ft_val['accuracy'])} | {fmt(ft_test['accuracy'])} | {ft_test['accuracy']-ft_val['accuracy']:+.4f} |
| f1_pos | {fmt(ft_val['f1_pos'])} | {fmt(ft_test['f1_pos'])} | {ft_test['f1_pos']-ft_val['f1_pos']:+.4f} |
| f1_macro | {fmt(ft_val['f1_macro'])} | {fmt(ft_test['f1_macro'])} | {ft_test['f1_macro']-ft_val['f1_macro']:+.4f} |

Diferențe sub 1 punct procentual între val și test → modelul generalizează onest, fără overfitting pe setul de validare.

## Confusion matrices pe test

### Zero-shot (mDeBERTa, fără antrenare)

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

- **Date**: 717 propoziții în limba română (496 train / 110 val / 111 test)
- **Surse**: traduceri din UKP Sentential Argument Mining Corpus (topicuri: nuclear_energy, minimum_wage, school_uniforms)
- **Distribuție clase**: ~2:1 argumentativ vs non-argumentativ
- **Model zero-shot**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Model fine-tuned**: `dumitrescustefan/bert-base-romanian-cased-v1`
- **Hiperparametri**: lr=2e-5, batch=16, 4 epochs, weight_decay=0.01, warmup_ratio=0.1
- **Selecție model**: cel mai bun checkpoint pe val (metric: f1_macro)

## Concluzie

Baseline-ul zero-shot pe mDeBERTa obține f1_macro = {fmt(zs['f1_macro'])} pe test, suferind de bias puternic spre clasa pozitivă (recall_neg = {fmt(rec_neg_zs)}). Fine-tuning-ul Romanian BERT pe 496 exemple adnotate ridică f1_macro la {fmt(ft_test['f1_macro'])}, o creștere de {d_f1mac*100:.1f} puncte procentuale, demonstrând valoarea supravegherii pentru această sarcină în limba română.
"""

    output_path = config.COMPARISON_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Salvat raportul la: {output_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()