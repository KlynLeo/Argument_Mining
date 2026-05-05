# Mihai relation pipeline

This module covers the SBERT and support/attack graph part of the project.

## Inputs from the full pipeline

After Tudor and Diana run their models, the input CSV should contain:

```csv
topic,sentence_ro,argument_pred,stance_pred
minimum wage,"Cresterea salariului minim ajuta familiile sarace.",argument,pro
minimum wage,"Unele firme ar putea reduce angajarile.",argument,contra
minimum wage,"Aceasta este o propozitie descriptiva.",no,neutral
```

Accepted `argument_pred` values include `argument`, `no`, `true`, `false`, `1`, `0`.
Accepted `stance_pred` values include `pro`, `contra`, `neutral`, `neutru`.

## Step 1: Generate candidate pairs and the baseline graph

If Tudor and Diana already exported their result files, first merge them into the graph input format:

```powershell
python data\prepare_data\integrate_predictions.py `
  --argument-input results\finetuned_predictions.csv `
  --stance-input results\finetuned_predictions_clasif2.csv `
  --output results\pipeline_predictions_for_graph.csv
```

This converts:

- Tudor/classifier 1: `predicted_label` 0/1 -> `argument_pred` no/argument;
- Diana/classifier 2: `predicted_label` 0/1 -> `stance_pred` pro/contra;
- missing stance predictions -> `neutral`.

Then build the graph from the merged file:

```powershell
python data\prepare_data\argument_graph.py `
  --input results\pipeline_predictions_for_graph.csv `
  --argument-col argument_pred `
  --stance-col stance_pred `
  --include-no-relation `
  --output-pairs results\pipeline_argument_pairs.csv `
  --output-edges results\pipeline_argument_edges.csv
```

What this does:

- keeps only argumentative `pro`/`contra` sentences;
- encodes each argument with multilingual SBERT;
- computes cosine similarity;
- creates candidate pairs above the similarity threshold;
- optionally samples low-similarity `no_relation` pairs;
- builds a baseline graph where same stance means `support` and opposite stance means `attack`.

`argument_edges.csv` is for Daniel's graph demo. `argument_pairs.csv` is for relation annotation/training.

## Step 2: Annotate relation pairs

Open `argument_pairs.csv` and fill the `relation_label` column with one of:

- `support`: the target reinforces or gives a compatible reason for the source;
- `attack`: the target contradicts, weakens, or opposes the source;
- `no_relation`: the two arguments are not meaningfully connected.

The `heuristic_relation` column is only a baseline suggestion. It should not be reported as gold labels.

## Step 3: Train and evaluate the relation classifier

```powershell
python data\prepare_data\relation_classifier.py `
  --input data\prepare_data\argument_pairs.csv `
  --label-col relation_label `
  --metrics-out data\prepare_data\relation_metrics.json `
  --model-out data\prepare_data\relation_classifier.joblib
```

This reports macro-F1 and support/attack macro-F1, satisfying the relation evaluation requirement once
`relation_label` contains manual labels.

For a smoke test only, not for final reporting, use:

```powershell
python data\prepare_data\relation_classifier.py `
  --input data\prepare_data\argument_pairs.csv `
  --label-col heuristic_relation `
  --metrics-out data\prepare_data\relation_metrics_heuristic.json `
  --no-save-model
```

## Step 4: Predict relations for Daniel's final graph

After a model is trained:

```powershell
python data\prepare_data\relation_classifier.py `
  --load-model data\prepare_data\relation_classifier.joblib `
  --predict-input data\prepare_data\argument_pairs.csv `
  --predictions-out data\prepare_data\relation_predictions.csv
```

Daniel can draw edges from `relation_predictions.csv` using `predicted_relation`.
