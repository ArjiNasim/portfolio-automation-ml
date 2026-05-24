# =============================================================================
# TP3 — Classification supervisée BUY / HOLD / SELL
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install scikit-learn xgboost lightgbm shap ta imbalanced-learn
# Input     : Companies_historical_data/*.csv
# Outputs   : outputs/classification_report.png + outputs/shap_*.png
# =============================================================================

# pip install ta imbalanced-learn  ← déjà dans requirements.txt

import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands

HIST_FOLDER = "Companies_historical_data"
OUT_DIR     = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 — Construction des labels et features (v1 et v2)
# =============================================================================
def preprocess_single_company(file_path):
    """
    Version v1 : Close brut + 9 features techniques.
    Labels : BUY(2) si rendement 20j > +5%, SELL(0) si < -5%, HOLD(1) sinon.
    """
    df = pd.read_csv(file_path, index_col=0)
    cr = df[["Close"]].copy()

    # Labels
    cr["Close Horizon"]  = cr["Close"].shift(-20)
    cr["horizon_return"] = (cr["Close Horizon"] - cr["Close"]) / cr["Close"]
    conditions = [(cr["horizon_return"] > 0.05), (cr["horizon_return"] < -0.05)]
    cr["label"] = np.select(conditions, [2, 0], default=1)

    # Features techniques
    cr["SMA_20"]               = SMAIndicator(cr["Close"], 20).sma_indicator()
    cr["EMA_20"]               = EMAIndicator(cr["Close"], 20).ema_indicator()
    cr["RSI_14"]               = RSIIndicator(cr["Close"], 14).rsi()
    macd_obj                   = MACD(cr["Close"])
    cr["MACD"]                 = macd_obj.macd()
    cr["MACD Signal"]          = macd_obj.macd_signal()
    bb                         = BollingerBands(cr["Close"])
    cr["Bollinger High"]       = bb.bollinger_hband()
    cr["Bollinger Low"]        = bb.bollinger_lband()
    cr["Rolling Volatility 20"] = cr["Close"].pct_change().rolling(20).std()
    cr["ROC_10"]               = ROCIndicator(cr["Close"], 10).roc()

    return cr.dropna()


def preprocess_single_company_v2(file_path):
    """
    Version v2 : features stationnaires (ratios, oscillateurs) + variable month.
    Supprime les prix absolus pour éviter le look-ahead bias.
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    cr = df[["Close"]].copy()

    cr["Close Horizon"]  = cr["Close"].shift(-20)
    cr["horizon_return"] = (cr["Close Horizon"] - cr["Close"]) / cr["Close"]
    conditions = [(cr["horizon_return"] > 0.05), (cr["horizon_return"] < -0.05)]
    cr["label"] = np.select(conditions, [2, 0], default=1)

    sma_20      = SMAIndicator(cr["Close"], 20).sma_indicator()
    ema_20      = EMAIndicator(cr["Close"], 20).ema_indicator()
    macd_obj    = MACD(cr["Close"])
    macd_raw    = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    bb          = BollingerBands(cr["Close"])
    bb_high     = bb.bollinger_hband()
    bb_low      = bb.bollinger_lband()
    bb_range    = bb_high - bb_low

    cr["RSI_14"]                 = RSIIndicator(cr["Close"], 14).rsi()
    cr["ROC_10"]                 = ROCIndicator(cr["Close"], 10).roc()
    cr["Rolling Volatility 20"]  = cr["Close"].pct_change().rolling(20).std()
    cr["price_vs_sma"]           = cr["Close"] / sma_20
    cr["price_vs_ema"]           = cr["Close"] / ema_20
    cr["bollinger_pct"]          = (cr["Close"] - bb_low) / bb_range
    cr["macd_normalized"]        = macd_raw    / cr["Close"]
    cr["macd_signal_normalized"] = macd_signal / cr["Close"]
    cr["SMA_20"] = sma_20 ; cr["EMA_20"] = ema_20
    cr["Bollinger High"] = bb_high ; cr["Bollinger Low"] = bb_low
    cr["month"] = cr.index.month

    return cr.dropna()


def build_global_pipeline(folder_path, version="v1"):
    """
    Concatène tous les DataFrames, standardise X et sépare train/test (stratifié).
    version : 'v1' (features brutes) ou 'v2' (features stationnaires + SMOTE).
    """
    preprocess_fn = preprocess_single_company if version == "v1" else preprocess_single_company_v2
    files   = glob.glob(f"{folder_path}/*.csv")
    all_dfs = []

    for f in files:
        try:
            all_dfs.append(preprocess_fn(f))
        except Exception as e:
            print(f"  ⚠️  Erreur sur {f}: {e}")

    if not all_dfs:
        raise RuntimeError("Aucun DataFrame chargé.")

    df_global = pd.concat(all_dfs, axis=0)
    print(f"  Dataset {version} : {df_global.shape[0]:,} lignes × {df_global.shape[1]} colonnes")
    print(f"  Distribution des labels :\n{df_global['label'].value_counts().sort_index().to_dict()}")

    Y = df_global["label"]
    cols_drop = ["label", "Close", "Close Horizon", "horizon_return",
                 "Weekly return", "Next Day Close",
                 "SMA_20", "EMA_20", "Bollinger High", "Bollinger Low"]
    X = df_global.drop(columns=[c for c in cols_drop if c in df_global.columns])

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 2 — Entraînement et évaluation des classifieurs
# =============================================================================
def train_and_evaluate_classifier(model, param_grid, X_train, X_test,
                                   y_train, y_test, model_name,
                                   compute_shap=False, out_dir=OUT_DIR):
    """
    GridSearchCV + classification report + SHAP (optionnel pour RF et XGBoost).
    """
    print(f"\n  {'='*55}")
    print(f"  {model_name}")
    print(f"  {'='*55}")

    if param_grid:
        gs = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        print(f"  Meilleurs paramètres : {gs.best_params_}")
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred,
                                target_names=["Sell (0)", "Hold (1)", "Buy (2)"]))
    report = classification_report(y_test, y_pred, output_dict=True)

    # SHAP — uniquement pour les arbres
    if compute_shap and model_name in ["Random Forest", "XGBoost"]:
        try:
            explainer   = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

            for cls_idx, cls_name in [(0, "Sell"), (2, "Buy")]:
                fig, ax = plt.subplots(figsize=(8, 5))
                sv = shap_values[cls_idx] if isinstance(shap_values, list) \
                                          else shap_values[:, :, cls_idx]
                shap.summary_plot(sv, X_test, feature_names=X_test.columns,
                                  show=False)
                plt.title(f"SHAP — Prédiction '{cls_name}' ({model_name})")
                plt.tight_layout()
                path = os.path.join(out_dir,
                    f"shap_{model_name.lower().replace(' ','_')}_{cls_name.lower()}.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  SHAP sauvegardé → {path}")
        except Exception as e:
            print(f"  ⚠️  SHAP impossible : {e}")

    return best_model, report


def extraire_tableau_synthese(reports_dict):
    """Construit un DataFrame récapitulatif des métriques clés."""
    rows = []
    for model_name, report in reports_dict.items():
        key_sell = next((k for k in report if "0" in str(k)), None)
        key_hold = next((k for k in report if "1" in str(k)), None)
        key_buy  = next((k for k in report if "2" in str(k)), None)
        rows.append({
            "Modèle":         model_name,
            "Accuracy":       round(report.get("accuracy", 0), 4),
            "Précision Buy":  round(report[key_buy]["precision"], 4) if key_buy else None,
            "Recall Buy":     round(report[key_buy]["recall"],    4) if key_buy else None,
            "Précision Sell": round(report[key_sell]["precision"],4) if key_sell else None,
            "Recall Sell":    round(report[key_sell]["recall"],   4) if key_sell else None,
            "F1 Hold":        round(report[key_hold]["f1-score"], 4) if key_hold else None,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Lancement
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TP3 — Classification BUY / HOLD / SELL")
    print("=" * 60)

    # ── Pipeline v1 ───────────────────────────────────────────────────────────
    print("\n[1/2] Pipeline v1 (features brutes)")
    X_train, X_test, y_train, y_test = build_global_pipeline(HIST_FOLDER, "v1")

    all_reports = {}

    _, all_reports["Random Forest"] = train_and_evaluate_classifier(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [10, 15]},
        X_train, X_test, y_train, y_test, "Random Forest", compute_shap=True
    )
    _, all_reports["XGBoost"] = train_and_evaluate_classifier(
        XGBClassifier(random_state=42, eval_metric="mlogloss"),
        {"n_estimators": [50, 100], "learning_rate": [0.1], "max_depth": [5, 7]},
        X_train, X_test, y_train, y_test, "XGBoost", compute_shap=True
    )
    _, all_reports["KNN"] = train_and_evaluate_classifier(
        KNeighborsClassifier(),
        {"n_neighbors": [5, 11, 15], "weights": ["uniform", "distance"]},
        X_train, X_test, y_train, y_test, "KNN"
    )
    _, all_reports["Logistic Regression"] = train_and_evaluate_classifier(
        LogisticRegression(random_state=42),
        {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"], "max_iter": [1000]},
        X_train, X_test, y_train, y_test, "Logistic Regression"
    )
    _, all_reports["SVM"] = train_and_evaluate_classifier(
        SVC(),
        {"C": [1.0], "kernel": ["rbf"]},
        X_train, X_test, y_train, y_test, "SVM"
    )

    df_v1 = extraire_tableau_synthese(all_reports)
    print("\n--- Tableau comparatif v1 ---")
    print(df_v1.to_string(index=False))

    # ── Pipeline v2 (SMOTE + features stationnaires) ──────────────────────────
    print("\n[2/2] Pipeline v2 (SMOTE + features stationnaires)")
    X_tr2, X_te2, y_tr2, y_te2 = build_global_pipeline(HIST_FOLDER, "v2")

    all_reports_v2 = {}
    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X_tr2, y_tr2)
    print(f"  Après SMOTE : {pd.Series(y_res).value_counts().sort_index().to_dict()}")

    for name, model, params in [
        ("Random Forest v2", RandomForestClassifier(random_state=42, class_weight="balanced"),
         {"n_estimators": [50, 100], "max_depth": [10, 15]}),
        ("XGBoost v2", XGBClassifier(random_state=42, eval_metric="mlogloss"),
         {"n_estimators": [50, 100], "learning_rate": [0.1], "max_depth": [5, 7]}),
        ("KNN v2", KNeighborsClassifier(),
         {"n_neighbors": [5, 11, 15], "weights": ["uniform", "distance"]}),
        ("Logistic Regression v2", LogisticRegression(random_state=42, class_weight="balanced"),
         {"C": [0.1, 1.0], "solver": ["lbfgs"], "max_iter": [1000]}),
        ("SVM v2", SVC(class_weight="balanced"),
         {"C": [1.0], "kernel": ["rbf"]}),
        ("LightGBM v2", LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
         {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [5, 7]}),
    ]:
        _, all_reports_v2[name] = train_and_evaluate_classifier(
            model, params, X_res, X_te2, y_res, y_te2, name
        )

    df_v2 = extraire_tableau_synthese(all_reports_v2)
    df_all = pd.concat([df_v1, df_v2], ignore_index=True)
    print("\n--- Tableau comparatif complet v1 + v2 ---")
    print(df_all.to_string(index=False))

    # Feature importances du meilleur modèle RF v2
    rf_v2 = RandomForestClassifier(random_state=42, class_weight="balanced",
                                    n_estimators=100, max_depth=15)
    rf_v2.fit(X_res, y_res)
    importances = pd.Series(rf_v2.feature_importances_,
                             index=X_tr2.columns).sort_values(ascending=False)
    print("\n--- Top 10 Feature Importances (RF v2) ---")
    print(importances.head(10).round(4).to_string())

    print("\n" + "=" * 60)
    print("  ✅ TP3 terminé. Figures SHAP dans outputs/")
    print("=" * 60)
