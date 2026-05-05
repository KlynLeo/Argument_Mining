import pathlib
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from pyvis.network import Network
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


APP_TITLE = "Argument Mining - Results Explorer"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _results_dir() -> pathlib.Path:
    return _repo_root() / "results"


@st.cache_data
def load_csv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _list_pipeline_files(results_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    files = {}
    for path in sorted(results_dir.glob("pipeline_argument_*.csv")):
        files[path.name] = path
    for path in sorted(results_dir.glob("pipeline_predictions_for_graph.csv")):
        files[path.name] = path
    return files


def _list_prediction_files(results_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    files = {}
    for path in sorted(results_dir.glob("finetuned_predictions*.csv")):
        files[path.name] = path
    for path in sorted(results_dir.glob("zero_shot_predictions*.csv")):
        files[path.name] = path
    for path in sorted(results_dir.glob("zero_shot_ro_bert_predictions.csv")):
        files[path.name] = path
    return files


def _truncate(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_graph(df: pd.DataFrame, label_by_sentence: bool) -> Optional[str]:
    if "source" not in df.columns or "target" not in df.columns:
        return None

    net = Network(height="600px", width="100%", directed=True, bgcolor="#0f172a", font_color="#e2e8f0")
    net.barnes_hut()

    def stance_color(value: str) -> str:
        if value == "pro":
            return "#22c55e"
        if value == "contra":
            return "#ef4444"
        return "#38bdf8"

    def relation_color(value: str) -> str:
        if value == "support":
            return "#22c55e"
        if value == "attack":
            return "#ef4444"
        return "#a3a3a3"

    for _, row in df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        source_label = source
        target_label = target

        if label_by_sentence:
            source_label = _truncate(str(row.get("source_sentence", source)))
            target_label = _truncate(str(row.get("target_sentence", target)))

        source_stance = str(row.get("source_stance", ""))
        target_stance = str(row.get("target_stance", ""))
        relation = str(row.get("relation", row.get("heuristic_relation", "")))

        if source not in net.node_ids:
            net.add_node(source, label=source_label, color=stance_color(source_stance))
        if target not in net.node_ids:
            net.add_node(target, label=target_label, color=stance_color(target_stance))

        weight = row.get("weight", row.get("similarity", 1.0))
        title = f"relation: {relation}<br>weight: {weight}"
        net.add_edge(source, target, value=float(weight), color=relation_color(relation), title=title)

    return net.generate_html()


def _resolve_label_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    label_col = "label" if "label" in df.columns else None
    if "predicted_label" in df.columns:
        pred_col = "predicted_label"
    elif "pred_label" in df.columns:
        pred_col = "pred_label"
    else:
        pred_col = None
    return label_col, pred_col


def _render_predictions_page(results_dir: pathlib.Path) -> None:
    st.subheader("Classification Predictions")
    prediction_files = _list_prediction_files(results_dir)
    if not prediction_files:
        st.warning("No prediction CSV files found in results/.")
        return

    file_name = st.sidebar.selectbox("Prediction CSV", list(prediction_files.keys()))
    df = load_csv(prediction_files[file_name])
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    label_col, pred_col = _resolve_label_columns(df)
    if label_col is None or pred_col is None:
        st.info("This file does not contain label and prediction columns.")
        st.dataframe(df, use_container_width=True)
        return

    if "topic" in df.columns:
        topics = ["All"] + sorted(df["topic"].dropna().unique().tolist())
        topic = st.sidebar.selectbox("Topic", topics, key="pred_topic")
        if topic != "All":
            df = df[df["topic"] == topic]

    max_rows = st.sidebar.slider("Max rows", min_value=50, max_value=2000, value=300, step=50, key="pred_rows")
    display_df = df.head(max_rows)

    st.dataframe(display_df, use_container_width=True)

    y_true = df[label_col].astype(str)
    y_pred = df[pred_col].astype(str)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Macro F1", f"{macro_f1:.3f}")
    col3.metric("Topics", df["topic"].nunique() if "topic" in df.columns else 1)
    col4.metric("Samples", len(df))

    st.subheader("Label Distribution")
    label_counts = df[label_col].astype(str).value_counts().sort_index()
    pred_counts = df[pred_col].astype(str).value_counts().sort_index()
    counts_df = pd.DataFrame({"gold": label_counts, "pred": pred_counts}).fillna(0)
    st.bar_chart(counts_df)

    st.subheader("Confusion Matrix")
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=[f"gold:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    st.dataframe(matrix_df, use_container_width=True)

    if "topic" in df.columns:
        st.subheader("Accuracy by Topic")
        topic_acc = (
            df.assign(correct=df[label_col].astype(str) == df[pred_col].astype(str))
            .groupby("topic")["correct"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(topic_acc)

    st.subheader("Misclassified Samples")
    errors = df[df[label_col].astype(str) != df[pred_col].astype(str)]
    if errors.empty:
        st.write("No misclassifications found.")
    else:
        columns = [c for c in ["text", "annotation", "topic", label_col, pred_col] if c in errors.columns]
        st.dataframe(errors[columns].head(50), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Explore pipeline outputs and classification predictions.")

    results_dir = _results_dir()
    if not results_dir.exists():
        st.error(f"Results folder not found at: {results_dir}")
        return

    page = st.sidebar.radio("Page", ["Pipeline Graph", "Predictions"], index=0)

    if page == "Predictions":
        _render_predictions_page(results_dir)
        return

    pipeline_files = _list_pipeline_files(results_dir)
    if not pipeline_files:
        st.warning("No pipeline CSV files found in results/.")
        return

    file_name = st.sidebar.selectbox("Pipeline CSV", list(pipeline_files.keys()))
    df = load_csv(pipeline_files[file_name])

    st.subheader(f"{file_name}")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    if "topic" in df.columns:
        topics = ["All"] + sorted(df["topic"].dropna().unique().tolist())
        topic = st.sidebar.selectbox("Topic", topics, key="pipeline_topic")
        if topic != "All":
            df = df[df["topic"] == topic]

    max_rows = st.sidebar.slider("Max rows", min_value=50, max_value=2000, value=300, step=50, key="pipeline_rows")
    display_df = df.head(max_rows)

    st.dataframe(display_df, use_container_width=True)

    if "source" in df.columns and "target" in df.columns:
        st.subheader("Argument Graph")
        label_by_sentence = st.checkbox("Label nodes by sentences", value=True)
        max_edges = st.slider("Max edges", min_value=50, max_value=1000, value=200, step=50)
        graph_df = df.head(max_edges)
        graph_html = _build_graph(graph_df, label_by_sentence)
        if graph_html:
            st.components.v1.html(graph_html, height=620, scrolling=True)
        else:
            st.info("Graph cannot be built for this file.")


if __name__ == "__main__":
    main()
