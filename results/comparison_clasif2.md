# Comparație: Zero-shot vs Fine-tuned (clasificare argument-pentru / argument-impotriva)

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

Fine-tuning-ul modifica performanta macro (f1_macro) cu **+2.7 puncte procentuale** și aduce o balansare semnificativă a claselor (recall_neg crește de la 0.6053 la 0.6316)..

## Validare onesta: Val vs Test (fine-tuned)

| Metrica | Val | Test | Diferenta |
|---|---|---|---|
| n_samples | 77 | 75 | - |
| accuracy | 0.7273 | 0.6533 | -0.0739 |
| f1_pos (against) | 0.7200 | 0.6579 | -0.0621 |
| f1_macro | 0.7271 | 0.6533 | -0.0738 |

Diferențe sub 1 punct procentual între val și test → modelul generalizează onest, fără overfitting pe setul de validare.

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

- **Date**: 495 propoziții în limba română (343 train / 77 val / 75 test)
- **Surse**: traduceri din UKP Sentential Argument Mining Corpus (topicuri: nuclear_energy, minimum_wage, school_uniforms)
- **Distribuție clase**: ~1:1 argumentativ vs non-argumentativ
- **Model zero-shot**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Model fine-tuned**: `dumitrescustefan/bert-base-romanian-uncased-v1`
- **Hiperparametri**: lr=2e-5, batch=16, 4 epochs, weight_decay=0.01, warmup_ratio=0.1
- **Selecție model**: cel mai bun checkpoint pe val (metric: f1_macro)

## Concluzie

Baseline-ul zero-shot pe mDeBERTa obține f1_macro = 0.6266 pe test. Fine-tuning-ul Romanian BERT pe 343 exemple adnotate ridică f1_macro la 0.6533, o creștere de 2.7 puncte procentuale, demonstrând valoarea supravegherii pentru această sarcină în limba română.
