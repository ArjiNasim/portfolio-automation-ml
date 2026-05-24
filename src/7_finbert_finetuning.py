# =============================================================================
# TP7 — Fine-tuning BERT et FinBERT pour l'analyse de sentiment financier
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install transformers datasets torch scikit-learn
# GPU recommandé (Google Colab T4 ou mieux).
# En CPU : l'entraînement prend ~2h par modèle.
#
# Datasets HuggingFace utilisés :
#   - zeroshot/twitter-financial-news-sentiment  (tweets financiers annotés)
#   - nickmuchi/financial-classification         (phrases économiques formelles)
#
# Outputs :
#   - ./bert-base-uncased_finetuned/    → BERT fine-tuné (accuracy 0.8645)
#   - ./ProsusAI_finbert_finetuned/     → FinBERT fine-tuné (accuracy 0.8581)
# =============================================================================

# pip install transformers datasets  ← déjà dans requirements.txt

from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
import numpy as np
import pandas as pd


# ── 1. Chargement et préparation des datasets ─────────────────────────────────
def load_and_prepare_datasets():
    """
    Charge et concatène deux datasets HuggingFace :
      - Twitter Financial News Sentiment  → 9 543 train + 2 388 validation
      - Financial Phrase Bank             → 4 551 train (split 80/20 créé)
    Retourne un DatasetDict {train, test} combiné de 13 183 / 3 299 exemples.
    """
    print("Chargement des datasets HuggingFace...")
    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds2 = load_dataset("nickmuchi/financial-classification")

    # Harmonisation du nom de colonne : "labels" → "label"
    ds2 = ds2.rename_column("labels", "label")

    ds1 = ds1.select_columns(["text", "label"])
    ds2 = ds2.select_columns(["text", "label"])

    # Création d'un split test pour ds2 (20%)
    ds2_split = ds2["train"].train_test_split(test_size=0.2, seed=42)

    combined = DatasetDict({
        "train": concatenate_datasets([ds1["train"],      ds2_split["train"]]),
        "test":  concatenate_datasets([ds1["validation"], ds2_split["test"]]),
    })

    print(f"  ✅ Train : {len(combined['train']):,} exemples")
    print(f"  ✅ Test  : {len(combined['test']):,} exemples")
    return combined


# ── 2. Métriques d'évaluation ────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Calcule accuracy, F1, precision, recall pour le Trainer."""
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# ── 3. Fine-tuning d'un modèle ───────────────────────────────────────────────
def train_model(model_name, dataset, batch_size=16, num_epochs=3):
    """
    Fine-tune un modèle Transformer sur le dataset financier.
    Sauvegarde les poids dans ./{model_name_clean}_finetuned/
    """
    print(f"\n{'='*60}")
    print(f"  Fine-tuning : {model_name}")
    print(f"{'='*60}")

    # Tokenizer et modèle pré-entraînés
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,               # Negative=0, Neutral=1, Positive=2
        ignore_mismatched_sizes=True
    )

    # Tokenisation (max_length=128 — adapté aux textes financiers courts)
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_train = dataset["train"].map(tokenize, batched=True)
    tokenized_test  = dataset["test"].map(tokenize, batched=True)
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch",  columns=["input_ids", "attention_mask", "label"])

    # Arguments d'entraînement
    model_name_clean = model_name.replace("/", "_")
    training_args = TrainingArguments(
        output_dir=f"./{model_name_clean}_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True,
        load_best_model_at_end=True,
    )

    # Trainer HuggingFace
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Entraînement + évaluation
    trainer.train()
    results = trainer.evaluate()

    # Classification report détaillé
    preds  = trainer.predict(tokenized_test)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    print(f"\n--- Classification Report ({model_name}) ---")
    print(classification_report(y_true, y_pred,
                                target_names=["Negative", "Neutral", "Positive"]))

    # Sauvegarde des poids (réutilisés directement en TP8)
    save_path = f"./{model_name_clean}_finetuned"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  ✅ Modèle sauvegardé → {save_path}/")

    return trainer, results


# ── Lancement ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TP7 — Fine-tuning BERT & FinBERT")
    print("=" * 60)

    dataset = load_and_prepare_datasets()

    # Fine-tuning BERT généraliste
    _, results_bert = train_model(
        model_name="bert-base-uncased",
        dataset=dataset,
        batch_size=16,
        num_epochs=3,
    )

    # Fine-tuning FinBERT (pré-entraîné sur données financières)
    _, results_finbert = train_model(
        model_name="ProsusAI/finbert",
        dataset=dataset,
        batch_size=16,
        num_epochs=3,
    )

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Tableau comparatif BERT vs FinBERT")
    print("=" * 60)
    df_cmp = pd.DataFrame({
        "Modèle":    ["BERT (bert-base-uncased)", "FinBERT (ProsusAI/finbert)"],
        "Accuracy":  [results_bert["eval_accuracy"],   results_finbert["eval_accuracy"]],
        "F1":        [results_bert["eval_f1"],          results_finbert["eval_f1"]],
        "Precision": [results_bert["eval_precision"],   results_finbert["eval_precision"]],
        "Recall":    [results_bert["eval_recall"],      results_finbert["eval_recall"]],
        "Eval Loss": [results_bert["eval_loss"],        results_finbert["eval_loss"]],
    })
    print(df_cmp.to_string(index=False))
    print()
    print("  Modèles sauvegardés dans :")
    print("    ./bert-base-uncased_finetuned/")
    print("    ./ProsusAI_finbert_finetuned/")
    print("  → Ces dossiers seront utilisés directement par le TP8.")
