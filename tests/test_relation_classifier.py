import pandas as pd

from data.prepare_data.relation_classifier import (
    normalise_relation_label,
    predict_relation_pairs,
    prepare_relation_features,
    train_and_evaluate,
)


def _relation_rows():
    examples = [
        ("support", "pro", "pro", 0.88, "Salariul minim ajuta familiile.", "Veniturile mai mari reduc saracia."),
        ("support", "contra", "contra", 0.82, "Costurile cresc pentru firme.", "Companiile pot reduce angajarile."),
        ("support", "pro", "pro", 0.79, "Uniformele reduc diferentele.", "Elevii nu mai concureaza prin haine."),
        ("support", "contra", "contra", 0.76, "Energia nucleara produce deseuri.", "Deseurile radioactive sunt riscante."),
        ("attack", "pro", "contra", 0.84, "Energia nucleara are emisii mici.", "Energia nucleara produce deseuri."),
        ("attack", "contra", "pro", 0.80, "Salariul minim distruge locuri.", "Salariul minim creste consumul."),
        ("attack", "pro", "contra", 0.78, "Uniformele cresc egalitatea.", "Uniformele limiteaza exprimarea."),
        ("attack", "contra", "pro", 0.74, "Reactoarele sunt prea scumpe.", "Centralele produc energie stabila."),
        ("no_relation", "pro", "contra", 0.12, "Salariul minim ajuta familiile.", "Reactoarele folosesc uraniu."),
        ("no_relation", "contra", "pro", 0.09, "Uniformele sunt costisitoare.", "Energia nucleara are emisii mici."),
        ("no_relation", "pro", "pro", 0.08, "Elevii se simt egali.", "Salariile cresc consumul."),
        ("no_relation", "contra", "contra", 0.06, "Deseurile nucleare sunt riscante.", "Angajarile pot scadea."),
    ]
    return [
        {
            "source": idx,
            "target": idx + 100,
            "topic": "test",
            "similarity": similarity,
            "relation_label": label,
            "source_stance": source_stance,
            "target_stance": target_stance,
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "relation": label,
        }
        for idx, (label, source_stance, target_stance, similarity, source_sentence, target_sentence) in enumerate(examples)
    ]


def test_normalise_relation_label():
    assert normalise_relation_label("supports") == "support"
    assert normalise_relation_label("contradiction") == "attack"
    assert normalise_relation_label("unrelated") == "no_relation"


def test_train_evaluate_and_predict_relation_classifier():
    df = pd.DataFrame(_relation_rows())

    model, metrics = train_and_evaluate(df, test_size=0.25, random_state=42, max_features=100)
    predictions = predict_relation_pairs(model, df.drop(columns=["relation"]))

    assert metrics["num_pairs"] == 12
    assert set(metrics["labels"]) == {"support", "attack", "no_relation"}
    assert "predicted_relation" in predictions.columns


def test_prepare_relation_features():
    features = prepare_relation_features(pd.DataFrame(_relation_rows()))

    assert list(features.columns) == ["pair_text", "source_stance", "target_stance", "similarity"]
    assert "[SEP]" in features.loc[0, "pair_text"]
