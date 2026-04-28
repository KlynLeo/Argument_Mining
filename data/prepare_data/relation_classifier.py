import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "argument_pairs.csv"
DEFAULT_METRICS = BASE_DIR / "relation_metrics.json"
DEFAULT_MODEL_OUT = BASE_DIR / "relation_classifier.joblib"
DEFAULT_PREDICTIONS_OUT = BASE_DIR / "relation_predictions.csv"

SUPPORT_LABELS = {"support", "supports", "same", "pro"}
ATTACK_LABELS = {"attack", "attacks", "contradiction", "contradicts", "opposes", "opposition", "contra"}
NO_RELATION_LABELS = {"no_relation", "none", "neutral", "neutru", "unrelated", "no", "0"}
RELATION_LABELS = ("support", "attack", "no_relation")


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalise_relation_label(value: object) -> str:
    label = _normalise_label(value)
    if label in SUPPORT_LABELS:
        return "support"
    if label in ATTACK_LABELS:
        return "attack"
    if label in NO_RELATION_LABELS:
        return "no_relation"
    raise ValueError(f"Unknown relation label: {value!r}")


def load_relation_pairs(path: Path | str, label_col: str = "relation_label") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}. Found: {list(df.columns)}")

    label_values = df[label_col].fillna("").astype(str).str.strip()
    labeled_df = df[label_values != ""].copy()
    if labeled_df.empty:
        raise ValueError(
            f"No labeled relation pairs found in {path}. Fill {label_col!r}, use "
            "--label-col heuristic_relation for a weak baseline, or run argument_graph.py "
            "with --weak-label-relations."
        )

    labeled_df["relation"] = labeled_df[label_col].map(normalise_relation_label)
    return labeled_df.reset_index(drop=True)


def prepare_relation_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "source_sentence",
        "target_sentence",
        "source_stance",
        "target_stance",
        "similarity",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for relation classification: {missing_columns}")

    features = df.copy()
    features["pair_text"] = (
        features["source_sentence"].fillna("").astype(str)
        + " [SEP] "
        + features["target_sentence"].fillna("").astype(str)
    )
    features["source_stance"] = features["source_stance"].fillna("unknown").astype(str)
    features["target_stance"] = features["target_stance"].fillna("unknown").astype(str)
    features["similarity"] = pd.to_numeric(features["similarity"], errors="coerce").fillna(0.0)
    return features[["pair_text", "source_stance", "target_stance", "similarity"]]


def build_relation_pipeline(max_features: int = 8000) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features), "pair_text"),
            ("stance", OneHotEncoder(handle_unknown="ignore"), ["source_stance", "target_stance"]),
            ("similarity", StandardScaler(), ["similarity"]),
        ]
    )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_relation_pairs(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    labels = df["relation"]
    label_counts = labels.value_counts()
    if len(label_counts) < 2:
        raise ValueError("At least two relation classes are required to train a classifier.")

    stratify = labels if label_counts.min() >= 2 else None
    features = prepare_relation_features(df)
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def relation_metrics(y_true: pd.Series, y_pred: list[str]) -> dict:
    present_labels = [label for label in RELATION_LABELS if label in set(y_true) | set(y_pred)]
    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "labels": present_labels,
        "macro_f1": f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0),
        "support_attack_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=[label for label in ("support", "attack") if label in present_labels],
            average="macro",
            zero_division=0,
        ),
        "per_label": {
            label: report[label]["f1-score"]
            for label in present_labels
            if label in report
        },
        "classification_report": report,
    }


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    max_features: int = 8000,
) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = split_relation_pairs(df, test_size=test_size, random_state=random_state)
    model = build_relation_pipeline(max_features=max_features)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = relation_metrics(y_test, predictions)
    metrics["num_pairs"] = int(len(df))
    metrics["num_train"] = int(len(x_train))
    metrics["num_test"] = int(len(x_test))
    metrics["label_distribution"] = {label: int(count) for label, count in df["relation"].value_counts().items()}
    return model, metrics


def load_candidate_pairs(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def predict_relation_pairs(model: Pipeline, pair_df: pd.DataFrame) -> pd.DataFrame:
    result_df = pair_df.copy()
    result_df["predicted_relation"] = model.predict(prepare_relation_features(pair_df))
    return result_df


def save_relation_predictions(prediction_df: pd.DataFrame, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved relation predictions: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a support/attack relation classifier")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV candidate pairs from argument_graph.py")
    parser.add_argument("--label-col", default="relation_label", help="Column containing manual relation labels")
    parser.add_argument("--test-size", type=float, default=0.25, help="Evaluation split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-features", type=int, default=8000, help="Maximum TF-IDF features")
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS), help="Output JSON metrics path")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT), help="Output trained model path")
    parser.add_argument("--no-save-model", action="store_true", help="Do not save the trained classifier")
    parser.add_argument("--load-model", default=None, help="Load an existing joblib model and skip training")
    parser.add_argument("--predict-input", default=None, help="Unlabeled candidate pairs to classify")
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS_OUT), help="Output CSV for predicted relations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.load_model:
        if not args.predict_input:
            raise ValueError("--load-model requires --predict-input")
        model = joblib.load(args.load_model)
        prediction_df = predict_relation_pairs(model, load_candidate_pairs(args.predict_input))
        save_relation_predictions(prediction_df, args.predictions_out)
        return

    relation_df = load_relation_pairs(args.input, label_col=args.label_col)
    model, metrics = train_and_evaluate(
        relation_df,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved relation metrics: {metrics_path}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Support/attack Macro-F1: {metrics['support_attack_macro_f1']:.4f}")

    if not args.no_save_model:
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved relation classifier: {model_path}")

    if args.predict_input:
        prediction_df = predict_relation_pairs(model, load_candidate_pairs(args.predict_input))
        save_relation_predictions(prediction_df, args.predictions_out)


if __name__ == "__main__":
    main()
