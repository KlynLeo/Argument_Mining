from __future__ import annotations

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

import config
import data_utils
import evaluate as eval_utils

def tokenize(df, tokenizer):
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=config.MAX_LENGTH), batched=True)
    return ds

def main():
    set_seed(config.SEED)

    train_df, val_df, test_df = data_utils.load_splits()
    
    tokenizer = AutoTokenizer.from_pretrained(config.RO_BERT_MODEL)
    train_ds = tokenize(train_df, tokenizer)
    val_ds = tokenize(val_df, tokenizer)
    test_ds = tokenize(test_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.RO_BERT_MODEL,
        num_labels=2,
        id2label=config.ID2LABEL,
        label2id={v: k for k, v in config.ID2LABEL.items()})

    args = TrainingArguments(
        output_dir=str(config.FINETUNED_MODEL_DIR/"checkpoints"),
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=config.SEED,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=eval_utils.compute_metrics_for_trainer,
    )
    
    trainer.train()
    
    val_preds = trainer.predict(val_ds)
    val_metrics = eval_utils.compute_metrics(val_preds.label_ids, np.argmax(val_preds.predictions, axis=-1))
    eval_utils.print_metrics("FINE-TUNING VAL", val_metrics)
    eval_utils.save_metrics(val_metrics, config.FINETUNED_VAL_METRICS_PATH)
    
    test_preds = trainer.predict(test_ds)
    y_pred = np.argmax(test_preds.predictions, axis=-1).tolist()
    test_metrics = eval_utils.compute_metrics(test_preds.label_ids, y_pred)
    eval_utils.print_metrics("FINE-TUNED (test)", test_metrics)
    eval_utils.save_metrics(test_metrics, config.FINETUNED_TEST_METRICS_PATH)

    test_df["predicted_label"] = y_pred
    test_df.to_csv(config.FINETUNED_PREDICTIONS_PATH, index=False)
    
    trainer.save_model(str(config.FINETUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(config.FINETUNED_MODEL_DIR))
    print(f"Model and tokenizer saved to {config.FINETUNED_MODEL_DIR}")
    
if __name__ == "__main__":
    main()
