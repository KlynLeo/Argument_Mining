# Comparație: Zero-shot vs Fine-tuned (clasificare argumentativ / non-argumentativ)

## Sinteza pe setul de test

| Metrica | Zero-shot (mDeBERTa) | Fine-tuned (RO-BERT) | Diferenta |
|---|---|---|---|
| n_samples | 75 | 75 | - |
| accuracy | 0.6267 | **0.6533** | +0.0267 |
| f1_pos | 0.6316 | **0.6579** | +0.0263 |
| **f1_macro** | 0.6266 | **0.6533** | **+0.0267** |
| precision_pos | 0.6154 | 0.6410 | +0.0256 |
| recall_pos | 0.6486 | 0.6757 | +0.0270 |
| recall_nonarg | 0.6053 | 0.6316 | +0.0263 |
| recall_arg | 0.6486 | 0.6757 | +0.0270 |

Fine-tuning-ul modifica performanta macro (f1_macro) cu **+2.7 puncte procentuale** pe test.

## Validare onesta: Val vs Test (fine-tuned)

| Metrica | Val | Test | Diferenta |
|---|---|---|---|
| n_samples | 77 | 75 | - |
| accuracy | 0.7273 | 0.6533 | -0.0739 |
| f1_pos (against) | 0.7200 | 0.6579 | -0.0621 |
| f1_macro | 0.7271 | 0.6533 | -0.0738 |

## Confusion matrices pe test

### Zero-shot (mDeBERTa)

|  | pred 0 (non-arg) | pred 1 (arg) |
|---|---|---|
| **true 0 (non-arg)** | 23 | 15 |
| **true 1 (arg)** | 13 | 24 |

### Fine-tuned (RO-BERT, 4 epochs)

|  | pred 0 (non-arg) | pred 1 (arg) |
|---|---|---|
| **true 0 (non-arg)** | 24 | 14 |
| **true 1 (arg)** | 12 | 25 |

## Classification reports detaliate (test)

### Zero-shot
```
                  precision    recall  f1-score   support

    argument_for     0.6389    0.6053    0.6216        38
argument_against     0.6154    0.6486    0.6316        37

        accuracy                         0.6267        75
       macro avg     0.6271    0.6270    0.6266        75
    weighted avg     0.6273    0.6267    0.6265        75

```

### Fine-tuned
```
                  precision    recall  f1-score   support

    argument_for     0.6667    0.6316    0.6486        38
argument_against     0.6410    0.6757    0.6579        37

        accuracy                         0.6533        75
       macro avg     0.6538    0.6536    0.6533        75
    weighted avg     0.6540    0.6533    0.6532        75

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
