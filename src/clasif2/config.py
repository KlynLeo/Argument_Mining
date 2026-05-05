from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "prepare_data" / "dataset_final.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN = "sentence_ro"
LABEL_COLUMN = "annotation"
SPLIT_COLUMN = "set"

LABEL2ID = {
    "Argument_for": 0,
    "Argument_against": 1,
}

ID2LABEL = {
    0: "argument_for",
    1: "argument_against",
}

RO_BERT_MODEL = "dumitrescustefan/bert-base-romanian-uncased-v1"
ZERO_SHOT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

FINETUNED_MODEL_DIR = MODELS_DIR / "ro_bert_argument_polarity"

SEED = 42
MAX_LENGTH = 128
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

ZERO_SHOT_LABELS = {
    0: "aceasta propozitie exprima un argument in favoarea",
    1: "aceasta propozitie exprima un argument impotriva",
}

ZERO_SHOT_HYPOTHESIS_TEMPLATE = "{}"

ZERO_SHOT_METRICS_PATH = RESULTS_DIR / "zero_shot_test_clasif2.json"
ZERO_SHOT_PREDICTIONS_PATH = RESULTS_DIR / "zero_shot_predictions_clasif2.csv"

FINETUNED_VAL_METRICS_PATH = RESULTS_DIR / "finetuned_val_clasif2.json"
FINETUNED_TEST_METRICS_PATH = RESULTS_DIR / "finetuned_test_clasif2.json"
FINETUNED_PREDICTIONS_PATH = RESULTS_DIR / "finetuned_predictions_clasif2.csv"

COMPARISON_PATH = RESULTS_DIR / "comparison_clasif2.md"


if __name__ == "__main__":
    print("=== CONFIG ===")
    print(f"PROJECT_ROOT       : {PROJECT_ROOT}")
    print(f"DATASET_PATH       : {DATASET_PATH}  (exists: {DATASET_PATH.exists()})")
    print(f"MODELS_DIR         : {MODELS_DIR}")
    print(f"RESULTS_DIR        : {RESULTS_DIR}")
    print(f"RO_BERT_MODEL      : {RO_BERT_MODEL}")
    print(f"ZERO_SHOT_MODEL    : {ZERO_SHOT_MODEL}")
    print(f"FINETUNED_MODEL_DIR: {FINETUNED_MODEL_DIR}")
    print(f"LABEL2ID           : {LABEL2ID}")
    print(f"SEED               : {SEED}")
