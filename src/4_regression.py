# =============================================================================
# TP4 — Régression et prédiction des prix à J+1
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install scikit-learn xgboost lightgbm ta numpy pandas
# Input     : Companies_historical_data/*.csv
# Outputs   : outputs/regression_*.png + affichage des métriques MSE/RMSE/MAE
# =============================================================================

# pip install ta  ← déjà dans requirements.txt

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands

HIST_FOLDER = "Companies_historical_data"
OUT_DIR     = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_DAYS    = 30   # Taille de la fenêtre glissante (30 closes)
N_FUTURE  = 10   # Horizon de prédiction multi-jours


# =============================================================================
# SECTION 1.1 — Construction du dataset par fenêtres glissantes (base)
# =============================================================================
def prepare_regression_data(file_path, n_days=N_DAYS, test_size=0.2):
    """
    Fenêtres glissantes sur les 30 derniers cours de clôture.
    ⚠️  Split TEMPOREL (pas aléatoire) pour éviter le look-ahead bias.
    ⚠️  MinMaxScaler fitté UNIQUEMENT sur le train.
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["Close"]].copy()

    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    def create_windows(data, n):
        X, Y = [], []
        for i in range(n, data.shape[0]):
            X.append(data[i - n:i].flatten())
            Y.append(data[i, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_windows(train_scaled, n_days)
    X_test,  y_test  = create_windows(test_scaled,  n_days)
    y_test_raw = test_df["Close"].iloc[n_days:].values

    return X_train, X_test, y_train, y_test, scaler, y_test_raw


# =============================================================================
# SECTION 1.2 — Pipeline de régression (5 modèles + GridSearchCV)
# =============================================================================
def run_regression_pipeline(X_train, X_test, y_train, y_test,
                             scaler, y_test_raw, company_name, save_prefix=""):
    """
    Entraîne 5 régresseurs avec GridSearchCV, affiche MSE/RMSE/MAE
    et génère le graphique vraies valeurs vs prédictions.
    """
    print(f"\n{'='*65}")
    print(f"  Pipeline de régression : {company_name}")
    print(f"{'='*65}")

    models_config = {
        "Régression Linéaire": (LinearRegression(), {}),
        "Random Forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                          {"n_estimators": [50, 100], "max_depth": [5, 10, None]}),
        "KNN":           (KNeighborsRegressor(),
                          {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}),
        "XGBoost":       (XGBRegressor(random_state=42, n_jobs=-1),
                          {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1],
                           "max_depth": [3, 5]}),
        "LightGBM":      (LGBMRegressor(random_state=42, n_jobs=-1),
                          {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                           "max_depth": [5, 7]}),
    }

    all_preds    = {}
    perf_records = []
    trained      = {}

    for name, (model, params) in models_config.items():
        print(f"\n  ▶ {name}...")
        if params:
            gs = GridSearchCV(model, params, cv=3,
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            print(f"    Params : {gs.best_params_}")
        else:
            best = model.fit(X_train, y_train)

        trained[name] = best
        preds_scaled  = best.predict(X_test)
        preds_prices  = scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)).flatten()
        all_preds[name] = preds_prices

        mse  = mean_squared_error(y_test_raw, preds_prices)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test_raw, preds_prices)
        print(f"    MSE={mse:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        perf_records.append({"Modèle": name, "MSE": mse, "RMSE": rmse, "MAE": mae})

    # Graphique
    colors = {"Régression Linéaire": "green", "Random Forest": "orange",
              "KNN": "purple", "XGBoost": "blue", "LightGBM": "brown"}
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_raw, color="red", linewidth=2, label="Valeurs réelles")
    for name, preds in all_preds.items():
        ax.plot(preds, color=colors[name], linestyle="--", label=name)
    ax.set_title(f"Comparaison des régresseurs — {company_name}")
    ax.set_xlabel("Jours (Période de Test)")
    ax.set_ylabel("Prix ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"regression_{save_prefix}_{company_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure sauvegardée → {path}")

    df_perf = pd.DataFrame(perf_records).set_index("Modèle")
    return df_perf, trained


# =============================================================================
# SECTION 1.3 — Version enrichie (30 closes + 9 features techniques du TP3)
# =============================================================================
def prepare_regression_data_v3(file_path, n_days=N_DAYS, test_size=0.2):
    """
    Enrichit les fenêtres glissantes avec les features stationnaires du TP3.
    Utilise deux scalers distincts : un pour Close (Y) et un pour les features (X).
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["Close"]].copy()

    sma20       = SMAIndicator(df["Close"], 20).sma_indicator()
    ema20       = EMAIndicator(df["Close"], 20).ema_indicator()
    macd_obj    = MACD(df["Close"])
    macd_raw    = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    bb          = BollingerBands(df["Close"])
    bb_h, bb_l  = bb.bollinger_hband(), bb.bollinger_lband()
    bb_range    = bb_h - bb_l

    df["RSI_14"]                 = RSIIndicator(df["Close"], 14).rsi()
    df["ROC_10"]                 = ROCIndicator(df["Close"], 10).roc()
    df["Volatility_20"]          = df["Close"].pct_change().rolling(20).std()
    df["price_vs_sma"]           = df["Close"] / sma20
    df["price_vs_ema"]           = df["Close"] / ema20
    df["bollinger_pct"]          = np.where(bb_range != 0,
                                       (df["Close"] - bb_l) / bb_range, 0.5)
    df["macd_normalized"]        = macd_raw    / df["Close"]
    df["macd_signal_normalized"] = macd_signal / df["Close"]
    df["month"]                  = df.index.month
    df = df.dropna()

    feature_cols  = [c for c in df.columns if c != "Close"]
    split_idx     = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    sc_close = MinMaxScaler() ; sc_feats = MinMaxScaler()
    tr_c = sc_close.fit_transform(train_df[["Close"]])
    tr_f = sc_feats.fit_transform(train_df[feature_cols])
    te_c = sc_close.transform(test_df[["Close"]])
    te_f = sc_feats.transform(test_df[feature_cols])

    def make_enriched(close_s, feats_s, n):
        X, Y = [], []
        for i in range(n, len(close_s)):
            X.append(np.concatenate([close_s[i - n:i].flatten(), feats_s[i]]))
            Y.append(close_s[i, 0])
        return np.array(X), np.array(Y)

    X_tr, y_tr = make_enriched(tr_c, tr_f, n_days)
    X_te, y_te = make_enriched(te_c, te_f, n_days)
    y_raw      = test_df["Close"].iloc[n_days:].values

    print(f"  v3 — X_train : {X_tr.shape}  "
          f"(30 closes + {X_tr.shape[1]-30} features)")
    return X_tr, X_te, y_tr, y_te, sc_close, y_raw


def predict_multistep(model, last_window, last_feats, scaler, n_future):
    """Prédiction autorégressive multi-jours (features figées à la dernière valeur connue)."""
    preds = []
    window = last_window.copy()
    for _ in range(n_future):
        x_in  = np.concatenate([window, last_feats]).reshape(1, -1)
        pred  = model.predict(x_in)[0]
        preds.append(pred)
        window = np.append(window[1:], pred)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# =============================================================================
# Lancement
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  TP4 — Régression et prédiction des prix à J+1")
    print("=" * 65)

    files = glob.glob(f"{HIST_FOLDER}/*.csv")
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {HIST_FOLDER}/")

    sample   = files[0]
    cname    = os.path.basename(sample).replace("_history.csv", "").replace("_", " ")
    print(f"\n  Entreprise sélectionnée : {cname}")

    # ── Version base ──────────────────────────────────────────────────────────
    print("\n[1/2] Version base (30 closes)")
    X_tr, X_te, y_tr, y_te, scaler, y_raw = prepare_regression_data(sample)
    df_perf, trained = run_regression_pipeline(
        X_tr, X_te, y_tr, y_te, scaler, y_raw, cname, save_prefix="base"
    )
    print("\n--- Tableau comparatif (base) ---")
    print(df_perf.sort_values("RMSE").round(4).to_string())

    # ── Version enrichie v3 ───────────────────────────────────────────────────
    print("\n[2/2] Version enrichie v3 (30 closes + 9 features)")
    X_tr3, X_te3, y_tr3, y_te3, sc3, y_raw3 = prepare_regression_data_v3(sample)
    df_perf3, trained3 = run_regression_pipeline(
        X_tr3, X_te3, y_tr3, y_te3, sc3, y_raw3, cname, save_prefix="v3"
    )
    print("\n--- Tableau comparatif (v3) ---")
    print(df_perf3.sort_values("RMSE").round(4).to_string())

    # ── Prédiction multi-jours ────────────────────────────────────────────────
    print(f"\n[Bonus] Prédiction itérative {N_FUTURE} jours")
    last_w    = X_tr3[-1][:N_DAYS]
    last_f    = X_tr3[-1][N_DAYS:]
    real_fut  = y_raw3[:N_FUTURE]
    records   = []

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real_fut, color="red", linewidth=2, label="Valeurs réelles")
    colors = {"Régression Linéaire": "green", "Random Forest": "orange",
              "KNN": "purple", "XGBoost": "blue", "LightGBM": "brown"}
    for mname, model in trained3.items():
        preds = predict_multistep(model, last_w, last_f, sc3, N_FUTURE)
        mae   = mean_absolute_error(real_fut, preds)
        rmse  = np.sqrt(mean_squared_error(real_fut, preds))
        records.append({"Modèle": mname,
                        "MAE (multi-j)":  round(mae, 4),
                        "RMSE (multi-j)": round(rmse, 4)})
        ax.plot(preds, color=colors[mname], linestyle="--", label=mname)

    ax.set_title(f"Prédiction itérative {N_FUTURE}j — {cname}")
    ax.set_xlabel("Jours") ; ax.set_ylabel("Prix ($)")
    ax.legend(fontsize=8)  ; ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"regression_multistep_{cname}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure sauvegardée → {path}")
    print(pd.DataFrame(records).set_index("Modèle").to_string())

    print("\n" + "=" * 65)
    print("  ✅ TP4 terminé. Figures dans outputs/")
    print("=" * 65)
