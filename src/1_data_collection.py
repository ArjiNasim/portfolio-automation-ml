# =============================================================================
# TP1 — Collecte et préparation des données financières
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install yfinance pandas
# Outputs   :
#   - data/ratios_financiers.csv          (41 entreprises × 13 ratios)
#   - Companies_historical_data/*.csv     (40 historiques de prix sur 5 ans)
# =============================================================================

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# ── Univers d'investissement (41 entreprises, 15 marchés) ─────────────────────
companies = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
    "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA",
    "NVIDIA": "NVDA", "Samsung": "005930.KS", "Tencent": "TCEHY",
    "Alibaba": "BABA", "IBM": "IBM", "Intel": "INTC",
    "Oracle": "ORCL", "Sony": "SONY", "Adobe": "ADBE",
    "Netflix": "NFLX", "AMD": "AMD", "Qualcomm": "QCOM",
    "Cisco": "CSCO", "JP Morgan": "JPM", "Goldman Sachs": "GS",
    "Visa": "V", "Johnson & Johnson": "JNJ", "Pfizer": "PFE",
    "ExxonMobil": "XOM", "ASML": "ASML.AS", "SAP": "SAP.DE",
    "Siemens": "SIE.DE", "Louis Vuitton (LVMH)": "MC.PA",
    "TotalEnergies": "TTE.PA", "Shell": "SHEL.L", "Baidu": "BIDU",
    "JD.com": "JD", "BYD": "BYDDY", "ICBC": "1398.HK",
    "Toyota": "TM", "SoftBank": "9984.T", "Nintendo": "NTDOY",
    "Hyundai": "HYMLF", "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
}

# ── 13 ratios financiers à collecter ──────────────────────────────────────────
RATIO_KEYS = [
    "forwardPE", "beta", "priceToBook", "priceToSales",
    "dividendYield", "trailingEps", "debtToEquity",
    "currentRatio", "quickRatio", "returnOnEquity",
    "returnOnAssets", "operatingMargins", "profitMargins",
]

# ── Fenêtre temporelle : 5 ans d'historique ───────────────────────────────────
END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

HIST_FOLDER = "Companies_historical_data"
DATA_FOLDER = "data"


# =============================================================================
# SECTION 1 — Collecte des ratios financiers
# =============================================================================
def collect_financial_ratios(companies, ratio_keys):
    """
    Pour chaque entreprise, récupère les 13 ratios financiers via yfinance.
    Utilise .get(key) pour éviter toute exception si un ratio est manquant.
    Retourne un DataFrame (41 lignes × 13 colonnes) avec Company en index.
    """
    ratios = {key: [] for key in ratio_keys}
    company_names = []

    for name, symbol in companies.items():
        print(f"  Extraction : {name} ({symbol})...")
        ticker = yf.Ticker(symbol)
        info   = ticker.info  # Un seul appel API par entreprise

        company_names.append(name)
        for key in ratio_keys:
            ratios[key].append(info.get(key))  # None si absent, pas d'exception

    df = pd.DataFrame(ratios, index=company_names)
    df.index.name = "Company"
    return df


# =============================================================================
# SECTION 2 — Collecte des historiques de prix (5 ans)
# =============================================================================
def collect_price_history(companies, start_date, end_date, folder):
    """
    Pour chaque entreprise, télécharge 5 ans de cours de clôture ajustés.
    Crée 3 colonnes : Close, Next Day Close, Rendement.
    Export CSV dans Companies_historical_data/
    """
    os.makedirs(folder, exist_ok=True)
    success, failed = 0, []

    for name, symbol in companies.items():
        try:
            print(f"  Téléchargement : {name} ({symbol})...")
            df = yf.download(symbol, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)

            if df.empty:
                print(f"    ⚠️  Données vides pour {name}")
                failed.append(name)
                continue

            # Correction MultiIndex (bug yfinance récent)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Construction des 3 colonnes
            df_hist = df[["Close"]].copy()
            df_hist["Next Day Close"] = df_hist["Close"].shift(-1)
            df_hist["Rendement"]      = (df_hist["Next Day Close"] - df_hist["Close"]) / df_hist["Close"]

            # Export CSV avec nom normalisé
            clean_name = name.replace(" ", "_").replace("&", "and")
            path = os.path.join(folder, f"{clean_name}_history.csv")
            df_hist.to_csv(path)
            print(f"    ✅ {name} → {path}")
            success += 1

        except Exception as e:
            print(f"    ❌ Erreur sur {name} : {e}")
            failed.append(name)

    return success, failed


# =============================================================================
# Lancement
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TP1 — Collecte et préparation des données financières")
    print("=" * 60)

    os.makedirs(DATA_FOLDER, exist_ok=True)

    # ── 1. Ratios financiers ───────────────────────────────────────────────────
    print("\n[1/2] Collecte des ratios financiers...")
    df_ratios = collect_financial_ratios(companies, RATIO_KEYS)
    ratios_path = os.path.join(DATA_FOLDER, "ratios_financiers.csv")
    df_ratios.to_csv(ratios_path)
    print(f"\n  ✅ {len(df_ratios)} entreprises exportées → {ratios_path}")
    print(f"  Aperçu :\n{df_ratios[['forwardPE','beta','returnOnEquity']].head(5)}")

    # ── 2. Historiques de prix ────────────────────────────────────────────────
    print(f"\n[2/2] Téléchargement des historiques ({START_DATE} → {END_DATE})...")
    ok, ko = collect_price_history(companies, START_DATE, END_DATE, HIST_FOLDER)
    print(f"\n  ✅ {ok} téléchargements réussis")
    if ko:
        print(f"  ❌ {len(ko)} échec(s) : {', '.join(ko)}")

    print("\n" + "=" * 60)
    print(f"  Output 1 : {ratios_path}")
    print(f"  Output 2 : {HIST_FOLDER}/ ({ok} fichiers CSV)")
    print("=" * 60)
