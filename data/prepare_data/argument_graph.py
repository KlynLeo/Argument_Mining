import argparse
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pyvis.network import Network

    _HAS_PYVIS = True
except ImportError:
    _HAS_PYVIS = False


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INPUT = BASE_DIR / "dataset_final.csv"
DEFAULT_OUTPUT_HTML = BASE_DIR / "argument_graph.html"
DEFAULT_OUTPUT_PNG = BASE_DIR / "argument_graph.png"
DEFAULT_OUTPUT_EDGES = BASE_DIR / "argument_edges.csv"
DEFAULT_OUTPUT_PAIRS = BASE_DIR / "argument_pairs.csv"

DEFAULT_ARGUMENT_COLUMNS = (
    "is_argument",
    "argument_pred",
    "argumentative",
    "predicted_argument",
    "argument_label",
)
DEFAULT_STANCE_COLUMNS = (
    "stance",
    "stance_pred",
    "predicted_stance",
    "stance_label",
)

PRO_LABELS = {"argument_for", "for", "pro", "support", "supports"}
CONTRA_LABELS = {"argument_against", "against", "contra", "con", "oppose", "opposes", "attack"}
NEUTRAL_LABELS = {"", "nan", "none", "null", "noargument", "no_argument", "not_argument", "non_argument", "neutral", "neutru"}
ARGUMENT_LABELS = PRO_LABELS | CONTRA_LABELS | {"argument", "argumentative", "yes", "true", "1"}
NON_ARGUMENT_LABELS = NEUTRAL_LABELS | {"nonargument", "no", "false", "0"}


def _normalise_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a UTF-8 CSV exported from data prep, annotation, or model inference."""
    return pd.read_csv(path, encoding="utf-8-sig")


def normalise_stance(value: object) -> str:
    label = _normalise_label(value)
    if label in PRO_LABELS:
        return "pro"
    if label in CONTRA_LABELS:
        return "contra"
    if label in NEUTRAL_LABELS:
        return "neutral"
    raise ValueError(f"Unknown stance label: {value!r}")


def normalise_argument_flag(value: object) -> bool:
    label = _normalise_label(value)
    if label in ARGUMENT_LABELS:
        return True
    if label in NON_ARGUMENT_LABELS:
        return False
    raise ValueError(f"Unknown argument label: {value!r}")


def prepare_argument_dataframe(
    df: pd.DataFrame,
    sentence_col: str = "sentence_ro",
    topic_col: str = "topic",
    annotation_col: str = "annotation",
    stance_col: str | None = None,
    argument_col: str | None = None,
) -> pd.DataFrame:
    """
    Return only argumentative rows with canonical columns used by the graph code.

    The current dataset uses gold labels in `annotation`. In the full pipeline, Tudor can
    provide an argument column and Diana can provide a stance column; this function accepts
    both shapes so the graph builder does not depend on gold labels.
    """
    if sentence_col not in df.columns:
        raise ValueError(f"Missing sentence column {sentence_col!r}. Found: {list(df.columns)}")

    working = df.copy()
    if topic_col not in working.columns:
        working[topic_col] = "document"

    stance_source = stance_col or _first_existing_column(working, DEFAULT_STANCE_COLUMNS)
    argument_source = argument_col or _first_existing_column(working, DEFAULT_ARGUMENT_COLUMNS)

    if argument_source:
        working["is_argument"] = working[argument_source].map(normalise_argument_flag)
    elif stance_source:
        inferred_stance = working[stance_source].map(normalise_stance)
        working["is_argument"] = inferred_stance.isin(["pro", "contra"])
    elif annotation_col in working.columns:
        working["is_argument"] = working[annotation_col].map(normalise_argument_flag)
    else:
        raise ValueError(
            "Cannot infer argumentative rows. Provide Tudor's argument column, Diana's "
            f"stance column, or an {annotation_col!r} column."
        )

    stance_source = stance_source or (annotation_col if annotation_col in working.columns else None)
    if not stance_source:
        raise ValueError(
            "Cannot infer stance. Provide a stance column from Diana's model or an "
            f"{annotation_col!r} column with Argument_for/Argument_against labels."
        )

    working["stance"] = "neutral"
    argument_mask = working["is_argument"]
    working.loc[argument_mask, "stance"] = working.loc[argument_mask, stance_source].map(normalise_stance)

    argument_df = working[working["is_argument"] & working["stance"].isin(["pro", "contra"])].copy()
    argument_df["sentence"] = argument_df[sentence_col]
    argument_df["topic"] = argument_df[topic_col]

    if "annotation" not in argument_df.columns:
        argument_df["annotation"] = argument_df["stance"].map({"pro": "Argument_for", "contra": "Argument_against"})

    return argument_df.reset_index(drop=True)


def encode_sentences(sentences: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for encoding. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    model = SentenceTransformer(model_name)
    return model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embeddings array, got shape {embeddings.shape}")
    return cosine_similarity(embeddings)


def infer_relation_from_stances(source_stance: str, target_stance: str) -> str:
    if source_stance == "neutral" or target_stance == "neutral":
        return "no_relation"
    return "support" if source_stance == target_stance else "attack"


def _pair_row(df: pd.DataFrame, similarity: float, source: int, target: int, relation: str) -> dict:
    source_row = df.loc[source]
    target_row = df.loc[target]
    return {
        "source": source,
        "target": target,
        "topic": source_row["topic"],
        "similarity": float(similarity),
        "heuristic_relation": relation,
        "relation_label": "",
        "source_stance": source_row["stance"],
        "target_stance": target_row["stance"],
        "source_sentence": source_row["sentence"],
        "target_sentence": target_row["sentence"],
    }


def generate_candidate_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.65,
    top_k: int = 3,
    group_by_topic: bool = True,
    include_no_relation: bool = False,
    no_relation_threshold: float = 0.35,
    max_no_relation_per_node: int = 1,
) -> pd.DataFrame:
    if len(df) != len(embeddings):
        raise ValueError(f"Rows ({len(df)}) and embeddings ({len(embeddings)}) must have the same length")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if max_no_relation_per_node < 0:
        raise ValueError("max_no_relation_per_node must be >= 0")

    df = df.reset_index(drop=True)
    sim = compute_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -1.0)

    groups = df.groupby("topic").groups.values() if group_by_topic else [range(len(df))]
    pair_rows = []
    seen_pairs = set()

    for group_indices in groups:
        indices = list(group_indices)
        for source in indices:
            candidates = [target for target in indices if target != source and sim[source, target] >= threshold]
            candidates = sorted(candidates, key=lambda target: sim[source, target], reverse=True)[:top_k]
            for target in candidates:
                pair_key = tuple(sorted((source, target)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                relation = infer_relation_from_stances(df.loc[source, "stance"], df.loc[target, "stance"])
                pair_rows.append(_pair_row(df, sim[source, target], pair_key[0], pair_key[1], relation))

        if include_no_relation and max_no_relation_per_node > 0:
            for source in indices:
                low_similarity_candidates = [
                    target
                    for target in indices
                    if target != source and sim[source, target] <= no_relation_threshold
                ]
                low_similarity_candidates = sorted(low_similarity_candidates, key=lambda target: sim[source, target])
                added = 0
                for target in low_similarity_candidates:
                    pair_key = tuple(sorted((source, target)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    pair_rows.append(_pair_row(df, sim[source, target], pair_key[0], pair_key[1], "no_relation"))
                    added += 1
                    if added >= max_no_relation_per_node:
                        break

    columns = [
        "source",
        "target",
        "topic",
        "similarity",
        "heuristic_relation",
        "relation_label",
        "source_stance",
        "target_stance",
        "source_sentence",
        "target_sentence",
    ]
    return pd.DataFrame(pair_rows, columns=columns)


def build_similarity_graph(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = 0.65,
    top_k: int = 3,
    group_by_topic: bool = True,
) -> nx.Graph:
    if len(df) != len(embeddings):
        raise ValueError(f"Rows ({len(df)}) and embeddings ({len(embeddings)}) must have the same length")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    graph = nx.Graph()
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        graph.add_node(
            idx,
            label=row["sentence"],
            sentence=row["sentence"],
            topic=row["topic"],
            stance=row["stance"],
            annotation=row.get("annotation", ""),
        )

    if top_k == 0 or len(df) < 2:
        return graph

    pair_df = generate_candidate_pairs(
        df,
        embeddings,
        threshold=threshold,
        top_k=top_k,
        group_by_topic=group_by_topic,
    )

    for _, row in pair_df.iterrows():
        graph.add_edge(
            int(row["source"]),
            int(row["target"]),
            weight=float(row["similarity"]),
            relation=row["heuristic_relation"],
        )

    return graph


def graph_edges_to_dataframe(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, attrs in graph.edges(data=True):
        source_attrs = graph.nodes[source]
        target_attrs = graph.nodes[target]
        rows.append(
            {
                "source": source,
                "target": target,
                "topic": source_attrs.get("topic", ""),
                "relation": attrs["relation"],
                "weight": attrs["weight"],
                "source_stance": source_attrs.get("stance", ""),
                "target_stance": target_attrs.get("stance", ""),
                "source_sentence": source_attrs.get("sentence", ""),
                "target_sentence": target_attrs.get("sentence", ""),
            }
        )
    return pd.DataFrame(rows)


def save_graph_pyvis(graph: nx.Graph, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.force_atlas_2based()

    for node, attrs in graph.nodes(data=True):
        color = "#4caf50" if attrs["stance"] == "pro" else "#f44336"
        title = f"<b>{attrs['stance'].upper()}</b><br>Topic: {attrs['topic']}<br>{attrs['sentence']}"
        net.add_node(node, label=str(node), title=title, color=color, shape="dot")

    for source, target, attrs in graph.edges(data=True):
        color = "#2196f3" if attrs["relation"] == "support" else "#ff9800"
        label = f"{attrs['relation']} ({attrs['weight']:.2f})"
        net.add_edge(source, target, title=label, color=color, width=2)

    net.write_html(str(output_path), open_browser=False, notebook=False)
    print(f"Saved interactive graph: {output_path}")


def save_graph_matplotlib(graph: nx.Graph, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42)

    pro_nodes = [node for node, data in graph.nodes(data=True) if data["stance"] == "pro"]
    contra_nodes = [node for node, data in graph.nodes(data=True) if data["stance"] == "contra"]

    nx.draw_networkx_nodes(graph, pos, nodelist=pro_nodes, node_color="#4caf50", node_size=140)
    nx.draw_networkx_nodes(graph, pos, nodelist=contra_nodes, node_color="#f44336", node_size=140)

    support_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["relation"] == "support"]
    attack_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["relation"] == "attack"]

    nx.draw_networkx_edges(graph, pos, edgelist=support_edges, edge_color="#2196f3", width=2)
    nx.draw_networkx_edges(graph, pos, edgelist=attack_edges, edge_color="#ff9800", style="dashed", width=2)
    nx.draw_networkx_labels(graph, pos, {node: str(node) for node in graph.nodes()}, font_size=8)

    plt.title("Argument similarity graph (SBERT + cosine similarity)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved static graph: {output_path}")


def save_edge_list(graph: nx.Graph, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_edges_to_dataframe(graph).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved edge list: {output_path}")


def save_candidate_pairs(pair_df: pd.DataFrame, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidate pairs: {output_path}")


def build_graph_from_dataframe(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    threshold: float = 0.65,
    top_k: int = 3,
    group_by_topic: bool = True,
    sentence_col: str = "sentence_ro",
    topic_col: str = "topic",
    annotation_col: str = "annotation",
    stance_col: str | None = None,
    argument_col: str | None = None,
) -> nx.Graph:
    argument_df = prepare_argument_dataframe(
        df,
        sentence_col=sentence_col,
        topic_col=topic_col,
        annotation_col=annotation_col,
        stance_col=stance_col,
        argument_col=argument_col,
    )
    embeddings = encode_sentences(argument_df["sentence"].tolist(), model_name)
    return build_similarity_graph(argument_df, embeddings, threshold=threshold, top_k=top_k, group_by_topic=group_by_topic)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an SBERT-based Romanian argument similarity graph")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV from data prep or model inference")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity threshold for edges")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=3, help="Max edges considered per node")
    parser.add_argument("--sentence-col", default="sentence_ro", help="Column containing Romanian sentences")
    parser.add_argument("--topic-col", default="topic", help="Column containing topic/document ids")
    parser.add_argument("--annotation-col", default="annotation", help="Gold annotation column, if available")
    parser.add_argument("--stance-col", default=None, help="Predicted stance column from Diana's model")
    parser.add_argument("--argument-col", default=None, help="Predicted argument column from Tudor's model")
    parser.add_argument("--allow-cross-topic", action="store_true", help="Allow edges between different topics/documents")
    parser.add_argument("--output-html", default=str(DEFAULT_OUTPUT_HTML), help="Output HTML file for pyvis graph")
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG), help="Output PNG file for static graph")
    parser.add_argument("--output-edges", default=str(DEFAULT_OUTPUT_EDGES), help="Output CSV edge list for the app pipeline")
    parser.add_argument("--output-pairs", default=str(DEFAULT_OUTPUT_PAIRS), help="Output CSV candidate pairs for relation annotation/training")
    parser.add_argument("--include-no-relation", action="store_true", help="Add low-similarity negative pairs for relation training")
    parser.add_argument("--no-relation-threshold", type=float, default=0.35, help="Maximum cosine similarity for negative pair sampling")
    parser.add_argument("--weak-label-relations", action="store_true", help="Copy the heuristic relation into relation_label for a baseline classifier demo")
    parser.add_argument("--no-html", action="store_true", help="Skip pyvis HTML output")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.input)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw_df = load_dataset(dataset_path)
    argument_df = prepare_argument_dataframe(
        raw_df,
        sentence_col=args.sentence_col,
        topic_col=args.topic_col,
        annotation_col=args.annotation_col,
        stance_col=args.stance_col,
        argument_col=args.argument_col,
    )
    print(f"Loaded {len(argument_df)} argumentative sentences from {dataset_path}")

    embeddings = encode_sentences(argument_df["sentence"].tolist(), args.model)
    pair_df = generate_candidate_pairs(
        argument_df,
        embeddings,
        threshold=args.threshold,
        top_k=args.top_k,
        group_by_topic=not args.allow_cross_topic,
        include_no_relation=args.include_no_relation,
        no_relation_threshold=args.no_relation_threshold,
    )
    if args.weak_label_relations:
        pair_df["relation_label"] = pair_df["heuristic_relation"]
    save_candidate_pairs(pair_df, args.output_pairs)

    graph = build_similarity_graph(
        argument_df,
        embeddings,
        threshold=args.threshold,
        top_k=args.top_k,
        group_by_topic=not args.allow_cross_topic,
    )
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    save_edge_list(graph, args.output_edges)

    if not args.no_html:
        if _HAS_PYVIS:
            try:
                save_graph_pyvis(graph, args.output_html)
            except Exception as exc:
                print(f"WARNING: pyvis could not save HTML output: {exc}")
        else:
            print("pyvis is not installed. Skipping HTML output.")

    if not args.no_png:
        save_graph_matplotlib(graph, args.output_png)


if __name__ == "__main__":
    main()
