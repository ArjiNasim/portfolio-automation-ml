# =============================================================================
# TP5 — Deep Learning : MLP, RNN, LSTM
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install tensorflow scikit-learn ta numpy pandas matplotlib
# Input     : Companies_historical_data/*.csv
# Outputs   : outputs/dl_*.png
# ⚠️  GPU recommandé (Google Colab T4). En CPU : ~15–30 min pour le grid search.
# =============================================================================

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands

HIST_FOLDER = "Companies_historical_data"
OUT_DIR     = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_DAYS   = 30
N_FUTURE = 10


# =============================================================================
# SECTION 1.1 — Dataset pour MLP (2D) et RNN/LSTM (3D)
# =============================================================================
def prepare_dl_dataset(file_path, n_days=N_DAYS, test_size=0.2):
    """
    Réutilise la logique de TP4 et retourne en plus les données en 3D
    (samples, timesteps, features) pour RNN/LSTM.
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["Close"]].copy()

    split_idx = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    def create_sequences(data, n):
        X, Y = [], []
        for i in range(n, len(data)):
            X.append(data[i - n:i, 0])
            Y.append(data[i, 0])
        return np.array(X), np.array(Y)

    X_tr_2d, y_train = create_sequences(train_scaled, n_days)
    X_te_2d, y_test  = create_sequences(test_scaled,  n_days)

    # 3D pour RNN/LSTM : (samples, timesteps, 1)
    X_tr_3d = X_tr_2d.reshape((X_tr_2d.shape[0], n_days, 1))
    X_te_3d = X_te_2d.reshape((X_te_2d.shape[0], n_days, 1))

    y_test_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    print(f"  MLP  — Train: {X_tr_2d.shape}  Test: {X_te_2d.shape}")
    print(f"  LSTM — Train: {X_tr_3d.shape}  Test: {X_te_3d.shape}")

    return X_tr_2d, X_te_2d, X_tr_3d, X_te_3d, y_train, y_test, scaler, y_test_raw


def prepare_dl_dataset_v2(file_path, n_days=N_DAYS, test_size=0.2):
    """
    Version enrichie : 30 closes + 9 features stationnaires du TP3.
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[["Close"]].copy()

    sma20       = SMAIndicator(df["Close"], 20).sma_indicator()
    ema20       = EMAIndicator(df["Close"], 20).ema_indicator()
    macd_obj    = MACD(df["Close"])
    macd_raw    = macd_obj.macd()
    macd_sig    = macd_obj.macd_signal()
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
    df["macd_normalized"]        = macd_raw / df["Close"]
    df["macd_signal_normalized"] = macd_sig  / df["Close"]
    df["month"]                  = df.index.month
    df = df.dropna()

    feature_cols = [c for c in df.columns if c != "Close"]
    n_features   = len(feature_cols)
    split_idx    = int(len(df) * (1 - test_size))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    sc_c = MinMaxScaler() ; sc_f = MinMaxScaler()
    tr_c = sc_c.fit_transform(train_df[["Close"]])
    tr_f = sc_f.fit_transform(train_df[feature_cols])
    te_c = sc_c.transform(test_df[["Close"]])
    te_f = sc_f.transform(test_df[feature_cols])

    def make_sequences(c, f, n):
        X2, X3, Y = [], [], []
        for i in range(n, len(c)):
            combined = np.concatenate([c[i - n:i].flatten(), f[i]])
            X2.append(combined)
            X3.append(combined.reshape(-1, 1))
            Y.append(c[i, 0])
        return np.array(X2), np.array(X3), np.array(Y)

    X_tr_2d, X_tr_3d, y_tr = make_sequences(tr_c, tr_f, n_days)
    X_te_2d, X_te_3d, y_te = make_sequences(te_c, te_f, n_days)
    y_raw                   = sc_c.inverse_transform(y_te.reshape(-1, 1)).flatten()
    input_dim               = n_days + n_features

    print(f"  v2 — MLP: {X_tr_2d.shape}  LSTM: {X_tr_3d.shape}  ({n_days}+{n_features} features)")
    return X_tr_2d, X_te_2d, X_tr_3d, X_te_3d, y_tr, y_te, sc_c, y_raw, input_dim


# =============================================================================
# SECTION 1.2 — Construction des architectures
# =============================================================================
def build_mlp_model(input_dim, hidden_dims=[64, 32], dropout_rate=0.2,
                    activation="relu", optimizer="adam", learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_dims[0], activation=activation,
                                     input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    for dim in hidden_dims[1:]:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate) if optimizer == "adam" \
          else tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model


def build_rnn_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.2,
                    activation="tanh", optimizer="adam", learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(
        hidden_dims[0], activation=activation,
        return_sequences=len(hidden_dims) > 1, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    for i, dim in enumerate(hidden_dims[1:]):
        ret_seq = i < len(hidden_dims) - 2
        model.add(tf.keras.layers.SimpleRNN(dim, activation=activation,
                                             return_sequences=ret_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate) if optimizer == "adam" \
          else tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model


def build_lstm_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.2,
                     activation="tanh", optimizer="adam", learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        hidden_dims[0], activation=activation,
        return_sequences=len(hidden_dims) > 1, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    for i, dim in enumerate(hidden_dims[1:]):
        ret_seq = i < len(hidden_dims) - 2
        model.add(tf.keras.layers.LSTM(dim, activation=activation,
                                        return_sequences=ret_seq))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate) if optimizer == "adam" \
          else tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(optimizer=opt, loss="mean_squared_error")
    return model


def build_hybrid_lstm_mlp(input_shape, lstm_units=64, dense_units=[32, 16],
                           dropout_rate=0.0, learning_rate=0.001):
    """LSTM pour la dynamique temporelle + MLP pour affiner la prédiction."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(lstm_units, activation="tanh",
                                    input_shape=input_shape, return_sequences=False))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="mean_squared_error")
    return model


# =============================================================================
# SECTION 1.2.2 — Entraînement unifié
# =============================================================================
def train_model(model_type, X_train, y_train, input_shape,
                hidden_dims=[64, 32], dropout_rate=0.2, activation="relu",
                optimizer="adam", learning_rate=0.001, epochs=50, batch_size=32):
    if model_type == "MLP":
        model = build_mlp_model(input_shape, hidden_dims, dropout_rate,
                                activation, optimizer, learning_rate)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, hidden_dims, dropout_rate,
                                "tanh", optimizer, learning_rate)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, hidden_dims, dropout_rate,
                                 "tanh", optimizer, learning_rate)
    else:
        raise ValueError(f"model_type inconnu : {model_type}")

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0)
    return model, history


def plot_training_history(history, model_type, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title(f"Courbe d'apprentissage — {model_type}")
    ax.set_xlabel("Epochs") ; ax.set_ylabel("MSE")
    ax.legend() ; ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# SECTION 1.2.3 — Prédiction et évaluation
# =============================================================================
def predict(model, X_test, y_test, scaler, model_type):
    """Prédit, inverse la normalisation et affiche MAE/RMSE."""
    y_pred_s = model.predict(X_test, verbose=0)
    y_pred   = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_real   = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mse  = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_real, y_pred)
    print(f"  {model_type:20} → MAE={mae:.4f}$  RMSE={rmse:.4f}$")
    return y_pred, mse, rmse, mae


# =============================================================================
# SECTION 1.2.4 — Grid Search (48 combinaisons × 3 modèles)
# =============================================================================
def run_grid_search(X_tr_2d, X_te_2d, X_tr_3d, X_te_3d,
                    y_train, y_test_raw, scaler, input_dim):
    """Grid Search sur hidden_dims, dropout, learning_rate, optimizer."""
    grid_params = {
        "hidden_dims":    [[64, 32], [128, 64]],
        "dropout_rate":   [0.0, 0.2],
        "learning_rate":  [0.01, 0.001],
        "optimizer_name": ["adam", "rmsprop"],
    }
    keys, values  = zip(*grid_params.items())
    combos        = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_metrics  = {"MLP": float("inf"), "RNN": float("inf"), "LSTM": float("inf")}
    best_objects  = {}
    records       = []

    for m_type in ["MLP", "RNN", "LSTM"]:
        X_tr = X_tr_2d if m_type == "MLP" else X_tr_3d
        X_te = X_te_2d if m_type == "MLP" else X_te_3d
        in_s = input_dim if m_type == "MLP" else (input_dim, 1)

        for p in combos:
            activation = "relu" if m_type == "MLP" else "tanh"
            try:
                if m_type == "MLP":
                    m = build_mlp_model(in_s, p["hidden_dims"], p["dropout_rate"],
                                        activation, p["optimizer_name"], p["learning_rate"])
                elif m_type == "RNN":
                    m = build_rnn_model(in_s, p["hidden_dims"], p["dropout_rate"],
                                        activation, p["optimizer_name"], p["learning_rate"])
                else:
                    m = build_lstm_model(in_s, p["hidden_dims"], p["dropout_rate"],
                                         activation, p["optimizer_name"], p["learning_rate"])

                m.fit(X_tr, y_train, epochs=20, batch_size=32, verbose=0)
                preds = scaler.inverse_transform(
                    m.predict(X_te, verbose=0).reshape(-1, 1)).flatten()

                mse  = mean_squared_error(y_test_raw, preds)
                rmse = np.sqrt(mse)
                mae  = mean_absolute_error(y_test_raw, preds)

                label = (f"{m_type} {p['hidden_dims']} "
                         f"drop:{p['dropout_rate']} "
                         f"lr:{p['learning_rate']} {p['optimizer_name']}")
                records.append({"Modèle": label, "Type": m_type,
                                "MSE": round(mse,4), "RMSE": round(rmse,4),
                                "MAE": round(mae,4)})

                if rmse < best_metrics[m_type]:
                    best_metrics[m_type] = rmse
                    best_objects[m_type] = {"model": m, "name": label,
                                            "preds": preds,
                                            "metrics": {"MSE": round(mse,4),
                                                        "RMSE": round(rmse,4),
                                                        "MAE": round(mae,4)}}
            except Exception as e:
                pass

    df_grid = pd.DataFrame(records)
    return df_grid, best_objects


# =============================================================================
# SECTION 1.3.1 — Prédiction multi-jours itérative DL
# =============================================================================
def predict_multistep_dl(model, model_type, last_sequence, scaler, n_future):
    preds = []
    seq   = last_sequence.copy()
    for _ in range(n_future):
        if model_type == "MLP":
            x_in   = seq.reshape(1, -1)
            pred   = model.predict(x_in, verbose=0)[0][0]
            new    = seq.copy()
            new[:-1] = seq[1:seq.shape[0]]
            new[-1]  = pred
            seq = new
        else:
            x_in   = seq.reshape(1, seq.shape[0], 1)
            pred   = model.predict(x_in, verbose=0)[0][0]
            new    = seq.copy()
            new[:-1] = seq[1:]
            new[-1]  = pred
            seq = new
        preds.append(pred)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# =============================================================================
# Lancement
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  TP5 — Deep Learning : MLP, RNN, LSTM")
    print("=" * 65)

    files = glob.glob(f"{HIST_FOLDER}/*.csv")
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {HIST_FOLDER}/")

    sample = files[0]
    cname  = os.path.basename(sample).replace("_history.csv","").replace("_"," ")
    print(f"\n  Entreprise : {cname}")

    # ── Dataset base ──────────────────────────────────────────────────────────
    print("\n[1/4] Préparation du dataset (base)")
    X2d, Xte2d, X3d, Xte3d, y_tr, y_te, scaler, y_raw = prepare_dl_dataset(sample)

    # ── Entraînement initial [64,32] ──────────────────────────────────────────
    print("\n[2/4] Entraînement initial [64,32] — 50 epochs")
    for m_type, X_tr_in, X_te_in, in_s in [
        ("MLP",  X2d,  Xte2d, N_DAYS),
        ("RNN",  X3d,  Xte3d, (N_DAYS, 1)),
        ("LSTM", X3d,  Xte3d, (N_DAYS, 1)),
    ]:
        m, hist = train_model(m_type, X_tr_in, y_tr, in_s,
                              hidden_dims=[64, 32], epochs=50)
        plot_training_history(hist, m_type,
            save_path=f"{OUT_DIR}/dl_learning_curve_{m_type.lower()}.png")
        predict(m, X_te_in, y_te, scaler, m_type)

    # ── Grid Search ───────────────────────────────────────────────────────────
    print("\n[3/4] Grid Search (48 combinaisons × 3 modèles)")
    df_grid, best_models = run_grid_search(X2d, Xte2d, X3d, Xte3d,
                                           y_tr, y_raw, scaler, N_DAYS)
    print("\n  Top 5 par modèle :")
    print(df_grid.groupby("Type").apply(
        lambda x: x.nsmallest(5, "RMSE")).reset_index(drop=True)
        [["Modèle","RMSE","MAE"]].to_string(index=False))

    # ── Dataset enrichi + modèle hybride ─────────────────────────────────────
    print("\n[4/4] Version enrichie + Hybride LSTM+MLP")
    (X2dv2, Xte2dv2, X3dv2, Xte3dv2,
     y_tr2, y_te2, sc2, y_raw2, idim2) = prepare_dl_dataset_v2(sample)

    df_grid2, best2 = run_grid_search(X2dv2, Xte2dv2, X3dv2, Xte3dv2,
                                      y_tr2, y_raw2, sc2, idim2)

    # Hybride
    m_hyb = build_hybrid_lstm_mlp((idim2, 1), lstm_units=64,
                                   dense_units=[32,16], dropout_rate=0.0)
    m_hyb.fit(X3dv2, y_tr2, epochs=20, batch_size=32, verbose=0)
    preds_hyb = sc2.inverse_transform(
        m_hyb.predict(Xte3dv2, verbose=0).reshape(-1,1)).flatten()
    rmse_hyb = np.sqrt(mean_squared_error(y_raw2, preds_hyb))
    mae_hyb  = mean_absolute_error(y_raw2, preds_hyb)
    print(f"\n  Hybride LSTM+MLP → RMSE={rmse_hyb:.4f}$  MAE={mae_hyb:.4f}$")
    best2["Hybrid"] = {"name": "Hybride LSTM(64)+MLP[32,16]",
                       "preds": preds_hyb,
                       "metrics": {"RMSE": round(rmse_hyb,4),
                                   "MAE":  round(mae_hyb,4)}}

    # Tableau final
    rows = []
    for m_type, cfg in {**best_models, **best2}.items():
        rows.append({"Modèle": cfg["name"], **cfg["metrics"]})
    df_final = pd.DataFrame(rows).set_index("Modèle").sort_values("RMSE")
    print("\n--- TABLEAU COMPARATIF FINAL ML vs DL ---")
    print(df_final.round(4).to_string())

    print("\n" + "=" * 65)
    print("  ✅ TP5 terminé. Figures dans outputs/")
    print("=" * 65)
