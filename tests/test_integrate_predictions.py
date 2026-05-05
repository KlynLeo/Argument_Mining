import pandas as pd

from data.prepare_data.integrate_predictions import build_graph_input


def test_build_graph_input_merges_argument_and_stance_predictions():
    argument_df = pd.DataFrame(
        {
            "text": ["arg pro", "not arg", "arg contra"],
            "topic": ["minimum wage", "minimum wage", "minimum wage"],
            "predicted_label": [1, 0, 1],
            "annotation": ["Argument_for", "NoArgument", "Argument_against"],
        }
    )
    stance_df = pd.DataFrame(
        {
            "text": ["arg pro", "arg contra"],
            "topic": ["minimum wage", "minimum wage"],
            "predicted_label": [0, 1],
        }
    )

    result = build_graph_input(argument_df, stance_df)

    assert result["sentence_ro"].tolist() == ["arg pro", "not arg", "arg contra"]
    assert result["argument_pred"].tolist() == ["argument", "no", "argument"]
    assert result["stance_pred"].tolist() == ["pro", "neutral", "contra"]


def test_build_graph_input_uses_neutral_when_stance_is_missing():
    argument_df = pd.DataFrame(
        {
            "text": ["arg without stance"],
            "topic": ["nuclear energy"],
            "predicted_label": [1],
        }
    )
    stance_df = pd.DataFrame(
        {
            "text": [],
            "topic": [],
            "predicted_label": [],
        }
    )

    result = build_graph_input(argument_df, stance_df)

    assert result["argument_pred"].tolist() == ["argument"]
    assert result["stance_pred"].tolist() == ["neutral"]
