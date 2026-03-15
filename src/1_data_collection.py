"""
TP1 – Financial Data Collection

In this script we perform two main tasks:

1) Scrape financial ratios for a set of companies using yfinance
2) Download 5 years of historical stock price data for each company

The outputs are:
- A CSV file containing financial ratios for all companies
- One CSV file per company containing historical stock data

This will serve as the raw dataset for the following machine learning steps.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1) Define project directories
# ============================================================

# Root directory of the project
BASE_DIR = Path(".")

# Directory where raw datasets will be stored
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Directory for historical stock data (one CSV per company)
HISTORICAL_DATA_DIR = RAW_DATA_DIR / "companies_historical_data"

# Create folders if they do not already exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) Define the companies we want to study
# ============================================================

# Dictionary mapping company names to stock tickers
companies: Dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Samsung": "005930.KS",
    "Tencent": "TCEHY",
    "Alibaba": "BABA",
    "IBM": "IBM",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Sony": "SONY",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Qualcomm": "QCOM",
    "Cisco": "CSCO",
    "JP Morgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "ExxonMobil": "XOM",
    "ASML": "ASML.AS",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Louis Vuitton (LVMH)": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Shell": "SHEL.L",
    "Baidu": "BIDU",
    "JD.com": "JD",
    "BYD": "BYDDY",
    "ICBC": "1398.HK",
    "Toyota": "TM",
    "SoftBank": "9984.T",
    "Nintendo": "NTDOY",
    "Hyundai": "HYMTF",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
}


# ============================================================
# 3) Define financial ratios we want to collect
# ============================================================

# These ratios will be extracted from yfinance
ratio_names: List[str] = [
    "forwardPE",
    "beta",
    "priceToBook",
    "priceToSales",
    "dividendYield",
    "trailingEps",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "returnOnEquity",
    "returnOnAssets",
    "operatingMargins",
    "profitMargins",
]


# ============================================================
# 4) Utility functions
# ============================================================

def sanitize_filename(name: str) -> str:
    """
    Convert a company name into a clean filename.
    This avoids problems with spaces or special characters.
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def safe_get(info: dict, key: str):
    """
    Safely retrieve a value from ticker.info.

    If the value does not exist we return NaN instead of raising an error.
    """
    value = info.get(key, np.nan)
    return np.nan if value is None else value


# ============================================================
# 5) Financial ratios scraping
# ============================================================

def fetch_company_ratios(symbol: str, ratio_list: List[str]) -> Dict[str, float]:
    """
    Retrieve financial ratios for one company using yfinance.
    """

    ticker = yf.Ticker(symbol)
    info = ticker.info

    ratios = {}

    for ratio in ratio_list:
        ratios[ratio] = safe_get(info, ratio)

    return ratios


def build_ratios_dataframe(
    companies_dict: Dict[str, str],
    ratio_list: List[str]
) -> pd.DataFrame:
    """
    Build a dataframe where:
    - rows = companies
    - columns = financial ratios
    """

    data = {}

    for company, ticker in companies_dict.items():

        print(f"Collecting ratios for {company} ({ticker})")

        try:
            data[company] = fetch_company_ratios(ticker, ratio_list)

        except Exception as e:
            print(f"Error for {company}: {e}")
            data[company] = {r: np.nan for r in ratio_list}

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Company"

    return df


# ============================================================
# 6) Historical price collection
# ============================================================

def download_price_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical stock prices from Yahoo Finance.
    """

    df = yf.download(symbol, start=start, end=end, progress=False)

    if df.empty:
        return df

    # Keep only the closing price
    df = df[["Close"]].copy()

    # Create the "Next Day Close" column
    df["Next Day Close"] = df["Close"].shift(-1)

    # Compute daily return
    df["Return"] = (df["Next Day Close"] - df["Close"]) / df["Close"]

    return df


def save_company_history(
    df: pd.DataFrame,
    company_name: str
):
    """
    Save historical price dataframe into a CSV file.
    """

    filename = sanitize_filename(company_name) + "_historical_data.csv"
    path = HISTORICAL_DATA_DIR / filename

    df.to_csv(path)


# ============================================================
# 7) Main execution pipeline
# ============================================================

def main():

    print("Starting financial data collection...")

    # --------------------------------------------------------
    # Step 1: collect financial ratios
    # --------------------------------------------------------

    ratios_df = build_ratios_dataframe(companies, ratio_names)

    ratios_output_path = RAW_DATA_DIR / "financial_ratios.csv"

    ratios_df.to_csv(ratios_output_path)

    print("Financial ratios saved.")

    print(ratios_df.head())

    # --------------------------------------------------------
    # Step 2: collect historical stock data
    # --------------------------------------------------------

    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print("Downloading 5 years of historical stock prices...")

    for company, ticker in companies.items():

        print(f"Downloading {company}")

        try:

            df = download_price_history(ticker, start_str, end_str)

            if not df.empty:
                save_company_history(df, company)

        except Exception as e:
            print(f"Error for {company}: {e}")

    print("TP1 completed successfully.")


# ============================================================
# 8) Run script
# ============================================================

if __name__ == "__main__":
    main()
