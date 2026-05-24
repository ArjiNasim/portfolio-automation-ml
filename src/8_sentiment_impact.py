# =============================================================================
# TP8 — Visualisation de l'impact des sentiments financiers sur les prix
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install transformers torch yfinance matplotlib pytz
#
# Inputs requis (générés par les TPs précédents) :
#   - companies_news/*.json               ← TP6 (news scrappées)
#   - ./ProsusAI_finbert_finetuned/       ← TP7 (modèle fine-tuné)
#   - ./bert-base-uncased_finetuned/      ← TP7 (modèle fine-tuné)
#
# Outputs :
#   - outputs/sentiment_vs_price_*.png    ← 1 figure par entreprise
#
# Fonctionnement :
#   1. get_texts_timestamps() : extrait textes + timestamps des JSONs TP6
#   2. get_sentiments()       : prédit le sentiment via FinBERT
#   3. align_timestamps()     : mappe les news sur les horaires de marché NYSE
#   4. plot_comparison()      : affiche prix horaire + points colorés côte à côte
# =============================================================================

# pip install pytz  ← déjà dans requirements.txt

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Mode sans affichage (pour exécution en script)

import yfinance as yf
import pandas as pd
import pytz
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from matplotlib.lines import Line2D


# ── Chemins des modèles ───────────────────────────────────────────────────────
FINBERT_BASE      = "ProsusAI/finbert"               # Sans fine-tuning
FINBERT_FINETUNED = "./ProsusAI_finbert_finetuned"   # Fine-tuné TP7
BERT_FINETUNED    = "./bert-base-uncased_finetuned"  # Fine-tuné TP7

NEWS_FOLDER   = "companies_news"
OUTPUTS_DIR   = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Entreprises analysées (celles avec le plus de news en TP6)
COMPANIES_TO_ANALYZE = {
    "Microsoft":    "MSFT",
    "Meta":         "META",
    "Amazon":       "AMZN",
    "NVIDIA":       "NVDA",
    "Goldman Sachs": "GS",
}

# Code couleur des sentiments
COLOR_MAP = {0: "red", 1: "gold", 2: "green"}
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ── 1. Extraction des textes et timestamps depuis les JSONs ──────────────────
def get_texts_timestamps(news_data):
    """
    Transforme le JSON d'une entreprise en deux listes parallèles :
      - news_texts      : textes (titre + description concaténés)
      - news_timestamps : timestamps convertis en timezone New York,
                          arrondis à l'heure pleine précédente
    """
    ny_tz = pytz.timezone("America/New_York")
    news_texts, news_timestamps = [], []

    for date, articles in news_data.items():
        for article in articles:
            title       = article.get("title", "")       or ""
            description = article.get("description", "") or ""
            text        = f"{title}. {description}".strip()
            if not text:
                continue

            published = article.get("publishedAt", "")
            if not published:
                continue

            # Conversion UTC → New York + arrondi à l'heure pleine
            dt_utc = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
            dt_utc = pytz.utc.localize(dt_utc)
            dt_ny  = dt_utc.astimezone(ny_tz)
            dt_ny  = dt_ny.replace(minute=0, second=0, microsecond=0)

            news_texts.append(text)
            news_timestamps.append(dt_ny)

    return news_texts, news_timestamps


# ── 2. Prédiction des sentiments ─────────────────────────────────────────────
def get_sentiments(model_path, texts):
    """
    Applique le modèle spécifié à chaque texte et retourne les prédictions.
    Labels : 0=Negative, 1=Neutral, 2=Positive
    Le tokenizer FinBERT est utilisé pour les deux modèles (même vocabulaire).
    """
    print(f"    Chargement du modèle : {model_path}")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model     = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    sentiments = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        sentiments.append(pred)

    return sentiments


# ── 3. Alignement sur les horaires de marché NYSE ────────────────────────────
def align_timestamps(timestamps):
    """
    Mappe chaque timestamp sur les horaires d'ouverture du marché :
      - 9h30 ≤ heure < 15h  → heure de publication (marché ouvert)
      - 15h  ≤ heure < 24h  → mappé à 15h le même jour (après clôture)
      - 0h   ≤ heure < 9h30 → mappé à 15h la veille (nuit)
    """
    aligned = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60

        if 9.5 <= hour < 15:
            aligned.append(ts)
        elif 15 <= hour < 24:
            aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
        else:
            veille = ts - timedelta(days=1)
            aligned.append(veille.replace(hour=15, minute=0, second=0, microsecond=0))

    return aligned


# ── 4. Visualisation côte à côte ─────────────────────────────────────────────
def plot_comparison(df, sentiments_a, sentiments_b, timestamps,
                    title_a, title_b, save_path=None):
    """
    Affiche deux graphiques côte à côte comparant les prédictions de sentiment
    de deux modèles sur la même courbe de prix horaire.
    Points : vert=Positive, or=Neutral, rouge=Negative.
    """
    ny_tz      = pytz.timezone("America/New_York")
    aligned_ts = align_timestamps(timestamps)

    # Conversion du DataFrame
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert(ny_tz)

    price_min = df["Close"].min()
    price_max = df["Close"].max()
    price_range = price_max - price_min

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, sentiments, title in zip(
        axes,
        [sentiments_a, sentiments_b],
        [title_a, title_b]
    ):
        # Courbe des prix
        ax.plot(df["Datetime"], df["Close"],
                color="steelblue", linewidth=1.5, label="Prix", zorder=1)

        # Groupement des news par timestamp aligné
        grouped = defaultdict(list)
        for ts, sent in zip(aligned_ts, sentiments):
            grouped[ts].append(sent)

        # Superposition des points colorés sur la courbe
        for ts, sents in grouped.items():
            mask       = df["Datetime"] == ts
            price_rows = df[mask]["Close"]
            if price_rows.empty:
                idx   = (df["Datetime"] - ts).abs().idxmin()
                price = df.loc[idx, "Close"]
            else:
                price = price_rows.values[0]

            for i, sent in enumerate(sents):
                # Décalage vertical pour distinguer les news simultanées
                offset = i * price_range * 0.01
                ax.scatter(ts, price + offset,
                           color=COLOR_MAP[sent], s=65, zorder=5,
                           edgecolors="white", linewidths=0.5)

        # Légende
        legend_elements = [
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="Prix"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="green", markersize=9, label="Positive"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="gold",  markersize=9, label="Neutral"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="red",   markersize=9, label="Negative"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix ($)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    ✅ Figure sauvegardée → {save_path}")
    else:
        plt.show()

    plt.close()


# ── Lancement principal ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TP8 — Sentiment vs Prix horaires")
    print("=" * 60)

    # Vérification des modèles TP7
    for model_dir in [FINBERT_FINETUNED, BERT_FINETUNED]:
        if not os.path.exists(model_dir):
            print(f"\n  ⚠️  Modèle introuvable : {model_dir}")
            print("     Lancer d'abord le TP7 pour générer les modèles fine-tunés.")
            print("     Commande : python src/7_finbert_finetuning.py")
            raise FileNotFoundError(f"Modèle manquant : {model_dir}")

    # Vérification du dossier de news
    if not os.path.exists(NEWS_FOLDER):
        raise FileNotFoundError(
            f"Dossier {NEWS_FOLDER}/ introuvable.\n"
            "Lancer d'abord le TP6 : python src/6_news_scraping.py"
        )

    for company, ticker_symbol in COMPANIES_TO_ANALYZE.items():
        print(f"\n{'─'*60}")
        print(f"  Analyse : {company} ({ticker_symbol})")
        print(f"{'─'*60}")

        # Chargement du JSON de news (TP6)
        json_path = os.path.join(NEWS_FOLDER, f"{company.replace(' ', '_')}.json")
        if not os.path.exists(json_path):
            print(f"  ⚠️  Pas de JSON pour {company} → ignoré.")
            continue

        with open(json_path, "r") as f:
            news_data = json.load(f)

        texts, timestamps = get_texts_timestamps(news_data)
        if not texts:
            print(f"  ⚠️  Aucun texte extrait pour {company} → ignoré.")
            continue
        print(f"  {len(texts)} articles chargés")

        # Historique des prix horaires (yfinance)
        ticker = yf.Ticker(ticker_symbol)
        df     = ticker.history(start="2026-05-01", interval="60m")
        df     = df.reset_index()
        print(f"  {len(df)} heures de données de prix")

        # Prédictions de sentiment — FinBERT base vs fine-tuné
        print("  Prédiction FinBERT base...")
        sentiments_base = get_sentiments(FINBERT_BASE, texts)

        print("  Prédiction FinBERT fine-tuné (TP7)...")
        sentiments_finetuned = get_sentiments(FINBERT_FINETUNED, texts)

        # Distribution des sentiments
        for label, sents in [("Base", sentiments_base), ("Fine-tuné", sentiments_finetuned)]:
            from collections import Counter
            c = Counter(sents)
            print(f"  {label:10} → Neg:{c[0]} | Neu:{c[1]} | Pos:{c[2]}")

        # Visualisation et sauvegarde
        save_path = os.path.join(OUTPUTS_DIR,
                                 f"sentiment_vs_price_{company.replace(' ', '_')}.png")
        plot_comparison(
            df=df,
            sentiments_a=sentiments_base,
            sentiments_b=sentiments_finetuned,
            timestamps=timestamps,
            title_a=f"{company} — FinBERT base",
            title_b=f"{company} — FinBERT fine-tuné (TP7)",
            save_path=save_path,
        )

    print("\n" + "=" * 60)
    print("  ✅ TP8 terminé.")
    print(f"  Figures disponibles dans : {OUTPUTS_DIR}/")
    print("=" * 60)
