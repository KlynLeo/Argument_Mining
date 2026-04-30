# Comparație: Zero-shot vs Fine-tuned (clasificare argumentativ / non-argumentativ)

## Sinteză pe setul de test

| Metrică | Zero-shot (mDeBERTa) | Fine-tuned (RO-BERT) | Diferență |
|---|---|---|---|
| n_samples | 111 | 111 | – |
| accuracy | 0.6396 | **0.7658** | +0.1261 |
| f1_pos | 0.7701 | **0.8354** | +0.0653 |
| **f1_macro** | 0.4684 | **0.7146** | **+0.2462** |
| precision_pos | 0.6768 | 0.7952 | +0.1184 |
| recall_pos | 0.8933 | 0.8800 | -0.0133 |
| recall_neg | 0.1111 | 0.5278 | +0.4167 |

Fine-tuning-ul îmbunătățește f1_macro cu **+24.6 puncte procentuale** și aduce o balansare semnificativă a claselor (recall_neg crește de la 0.1111 la 0.5278).

## Validare onestă: Val vs Test (fine-tuned)

| Metrică | Val | Test | Diferență |
|---|---|---|---|
| n_samples | 110 | 111 | – |
| accuracy | 0.7727 | 0.7658 | -0.0070 |
| f1_pos | 0.8428 | 0.8354 | -0.0073 |
| f1_macro | 0.7165 | 0.7146 | -0.0019 |

Diferențe sub 1 punct procentual între val și test → modelul generalizează onest, fără overfitting pe setul de validare.

## Confusion matrices pe test

### Zero-shot (mDeBERTa, fără antrenare)

|  | pred 0 (non-arg) | pred 1 (arg) |
|---|---|---|
| **true 0 (non-arg)** | 4 | 32 |
| **true 1 (arg)** | 8 | 67 |

### Fine-tuned (RO-BERT, 4 epochs)

|  | pred 0 (non-arg) | pred 1 (arg) |
|---|---|---|
| **true 0 (non-arg)** | 19 | 17 |
| **true 1 (arg)** | 9 | 66 |

## Classification reports detaliate (test)

### Zero-shot
```
                   precision    recall  f1-score   support

non-argumentative     0.3333    0.1111    0.1667        36
    argumentative     0.6768    0.8933    0.7701        75

         accuracy                         0.6396       111
        macro avg     0.5051    0.5022    0.4684       111
     weighted avg     0.5654    0.6396    0.5744       111

```

### Fine-tuned
```
                   precision    recall  f1-score   support

non-argumentative     0.6786    0.5278    0.5938        36
    argumentative     0.7952    0.8800    0.8354        75

         accuracy                         0.7658       111
        macro avg     0.7369    0.7039    0.7146       111
     weighted avg     0.7574    0.7658    0.7571       111

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

Baseline-ul zero-shot pe mDeBERTa obține f1_macro = 0.4684 pe test, suferind de bias puternic spre clasa pozitivă (recall_neg = 0.1111). Fine-tuning-ul Romanian BERT pe 496 exemple adnotate ridică f1_macro la 0.7146, o creștere de 24.6 puncte procentuale, demonstrând valoarea supravegherii pentru această sarcină în limba română.
