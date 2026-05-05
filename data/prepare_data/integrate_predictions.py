import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARGUMENT_INPUT = BASE_DIR / "results" / "finetuned_predictions.csv"
DEFAULT_STANCE_INPUT = BASE_DIR / "results" / "finetuned_predictions_clasif2.csv"
DEFAULT_OUTPUT = BASE_DIR / "results" / "pipeline_predictions_for_graph.csv"

ARGUMENT_LABELS = {
    0: "no",
    1: "argument",
}

STANCE_LABELS = {
    0: "pro",
    1: "contra",
}


def load_predictions(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def normalise_prediction_id(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction label must be 0 or 1, got {value!r}") from exc


def build_graph_input(
    argument_df: pd.DataFrame,
    stance_df: pd.DataFrame,
    text_col: str = "text",
    topic_col: str = "topic",
    prediction_col: str = "predicted_label",
) -> pd.DataFrame:
    required_columns = {text_col, topic_col, prediction_col}
    missing_argument = required_columns - set(argument_df.columns)
    missing_stance = required_columns - set(stance_df.columns)
    if missing_argument:
        raise ValueError(f"Argument predictions missing columns: {sorted(missing_argument)}")
    if missing_stance:
        raise ValueError(f"Stance predictions missing columns: {sorted(missing_stance)}")

    arguments = argument_df.copy()
    stances = stance_df[[text_col, topic_col, prediction_col]].copy()
    stances = stances.rename(columns={prediction_col: "stance_prediction_id"})

    merged = arguments.merge(stances, on=[text_col, topic_col], how="left")
    merged["argument_prediction_id"] = merged[prediction_col].map(normalise_prediction_id)
    merged["argument_pred"] = merged["argument_prediction_id"].map(ARGUMENT_LABELS)

    merged["stance_pred"] = "neutral"
    has_stance = merged["stance_prediction_id"].notna()
    merged.loc[has_stance, "stance_pred"] = (
        merged.loc[has_stance, "stance_prediction_id"].map(normalise_prediction_id).map(STANCE_LABELS)
    )
    merged.loc[merged["argument_pred"] == "no", "stance_pred"] = "neutral"

    output = pd.DataFrame(
        {
            "topic": merged[topic_col],
            "sentence_ro": merged[text_col],
            "argument_pred": merged["argument_pred"],
            "stance_pred": merged["stance_pred"],
            "argument_prediction_id": merged["argument_prediction_id"],
            "stance_prediction_id": merged["stance_prediction_id"],
        }
    )

    if "annotation" in merged.columns:
        output["gold_annotation"] = merged["annotation"]

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Tudor/Diana predictions into Mihai graph input")
    parser.add_argument("--argument-input", default=str(DEFAULT_ARGUMENT_INPUT), help="clasif1 prediction CSV")
    parser.add_argument("--stance-input", default=str(DEFAULT_STANCE_INPUT), help="clasif2 prediction CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="merged CSV for argument_graph.py")
    parser.add_argument("--text-col", default="text", help="sentence column in classifier outputs")
    parser.add_argument("--topic-col", default="topic", help="topic column in classifier outputs")
    parser.add_argument("--prediction-col", default="predicted_label", help="prediction column in classifier outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argument_df = load_predictions(args.argument_input)
    stance_df = load_predictions(args.stance_input)
    output_df = build_graph_input(
        argument_df,
        stance_df,
        text_col=args.text_col,
        topic_col=args.topic_col,
        prediction_col=args.prediction_col,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged graph input: {output_path}")
    print(f"Rows: {len(output_df)}")
    print("argument_pred counts:")
    print(output_df["argument_pred"].value_counts().to_string())
    print("stance_pred counts:")
    print(output_df["stance_pred"].value_counts().to_string())


if __name__ == "__main__":
    main()
