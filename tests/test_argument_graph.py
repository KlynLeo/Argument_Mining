import numpy as np
import pandas as pd

from data.prepare_data.argument_graph import (
    build_similarity_graph,
    generate_candidate_pairs,
    infer_relation_from_stances,
    prepare_argument_dataframe,
)


def test_prepare_argument_dataframe_accepts_gold_annotations():
    df = pd.DataFrame(
        {
            "topic": ["minimum wage", "minimum wage", "minimum wage"],
            "sentence_ro": ["pro", "contra", "neutral"],
            "annotation": ["Argument_for", "Argument_against", "NoArgument"],
        }
    )

    result = prepare_argument_dataframe(df)

    assert result["sentence"].tolist() == ["pro", "contra"]
    assert result["stance"].tolist() == ["pro", "contra"]


def test_prepare_argument_dataframe_accepts_pipeline_predictions():
    df = pd.DataFrame(
        {
            "topic": ["nuclear energy", "nuclear energy", "nuclear energy"],
            "sentence_ro": ["a", "b", "c"],
            "argument_pred": ["argument", "argument", "no"],
            "stance_pred": ["pro", "contra", "neutral"],
        }
    )

    result = prepare_argument_dataframe(df, stance_col="stance_pred", argument_col="argument_pred")

    assert result["sentence"].tolist() == ["a", "b"]
    assert result["stance"].tolist() == ["pro", "contra"]


def test_similarity_graph_stays_inside_topic_by_default():
    df = pd.DataFrame(
        {
            "topic": ["topic_a", "topic_b"],
            "sentence": ["same idea a", "same idea b"],
            "stance": ["pro", "pro"],
            "annotation": ["Argument_for", "Argument_for"],
        }
    )
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    graph = build_similarity_graph(df, embeddings, threshold=0.5, top_k=1)

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0


def test_candidate_pairs_export_support_attack_and_no_relation_rows():
    df = pd.DataFrame(
        {
            "topic": ["topic_a", "topic_a", "topic_a", "topic_a"],
            "sentence": ["pro one", "pro two", "contra one", "unrelated pro"],
            "stance": ["pro", "pro", "contra", "pro"],
            "annotation": ["Argument_for", "Argument_for", "Argument_against", "Argument_for"],
        }
    )
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]])

    pairs = generate_candidate_pairs(
        df,
        embeddings,
        threshold=0.5,
        top_k=2,
        include_no_relation=True,
        no_relation_threshold=0.1,
        max_no_relation_per_node=1,
    )

    assert {"support", "attack", "no_relation"}.issubset(set(pairs["heuristic_relation"]))


def test_infer_relation_from_stances():
    assert infer_relation_from_stances("pro", "pro") == "support"
    assert infer_relation_from_stances("contra", "pro") == "attack"
    assert infer_relation_from_stances("neutral", "pro") == "no_relation"
