# =============================================================================
# TP6 — Scraping de news financières via NewsAPI
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install requests pandas
# Clé API   : créer un compte sur https://newsapi.org (gratuit)
#             puis définir la variable d'environnement NEWS_API_KEY
#             Linux/Mac : export NEWS_API_KEY="votre_cle"
#             Windows   : set NEWS_API_KEY=votre_cle
# =============================================================================

import requests
import json
import os
import time
import pandas as pd
from datetime import datetime, timedelta

# ── Clé API (depuis variable d'environnement, jamais en dur dans le code) ────
API_KEY = os.environ.get("NEWS_API_KEY", "")
if not API_KEY:
    raise ValueError(
        "Clé NewsAPI manquante.\n"
        "Définir la variable d'environnement NEWS_API_KEY avant de lancer ce script.\n"
        "Exemple : export NEWS_API_KEY=votre_cle"
    )

NEWS_FOLDER = "companies_news"
os.makedirs(NEWS_FOLDER, exist_ok=True)

# ── Univers d'investissement (41 entreprises) ─────────────────────────────────
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

# ── Sélection des sources selon la zone géographique ─────────────────────────
ASIAN_SUFFIXES    = [".KS", ".T", "TCEHY", "BABA", "BIDU", "JD",
                     "BYDDY", "1398.HK", "NTDOY", "HYMLF", "RELIANCE.NS", "TCS.NS"]
EUROPEAN_SUFFIXES = [".PA", ".L", ".DE", ".AS"]

SOURCES_US     = ("financial-post,the-wall-street-journal,bloomberg,"
                  "reuters,the-economist,business-insider,fortune,forbes,"
                  "cnbc,the-verge,techcrunch,bbc-news,the-guardian-uk,"
                  "associated-press")
SOURCES_EUROPE = ("bbc-news,the-guardian-uk,reuters,the-economist,bloomberg,"
                  "le-monde,handelsblatt,der-spiegel,les-echos,el-mundo,"
                  "fortune,cnbc,associated-press")
SOURCES_ASIA   = ("reuters,bloomberg,bbc-news,al-jazeera-english,"
                  "associated-press,the-economist,cnbc,fortune,"
                  "the-japan-times,south-china-morning-post,"
                  "hindustan-times,channel-news-asia")

ASIAN_COMPANIES    = [n for n, t in companies.items()
                      if any(s in t for s in ASIAN_SUFFIXES)]
EUROPEAN_COMPANIES = [n for n, t in companies.items()
                      if any(t.endswith(s) for s in EUROPEAN_SUFFIXES)]

# Noms de recherche alternatifs pour certaines entreprises
SEARCH_NAMES = {
    "Louis Vuitton (LVMH)":        "LVMH",
    "JP Morgan":                    "JPMorgan",
    "Johnson & Johnson":            "Johnson Johnson",
    "Tata Consultancy Services":    "TCS",
    "Reliance Industries":          "Reliance",
    "JD.com":                       "JD.com",
}


def get_search_name(company):
    return SEARCH_NAMES.get(company, company)


def load_existing_news(company_name):
    """Charge le JSON existant pour une entreprise (mise à jour incrémentale)."""
    path = os.path.join(NEWS_FOLDER, f"{company_name.replace(' ', '_')}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_news_by_date(company_name, days_back=10):
    """
    Scrape les news d'une entreprise via NewsAPI.
    - Sélection automatique des sources selon la zone géographique.
    - Déduplication par titre (mise à jour incrémentale).
    - Export JSON dans companies_news/
    """
    url       = "https://newsapi.org/v2/everything"
    last_day  = datetime.today().strftime("%Y-%m-%d")
    first_day = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Sources adaptées à la zone géographique
    if company_name in ASIAN_COMPANIES:
        sources = SOURCES_ASIA
    elif company_name in EUROPEAN_COMPANIES:
        sources = SOURCES_EUROPE
    else:
        sources = SOURCES_US

    params = {
        "sources":  sources,
        "q":        get_search_name(company_name),
        "apiKey":   API_KEY,
        "language": "en",
        "pageSize": 100,
        "from":     first_day,
        "to":       last_day,
        "sortBy":   "relevancy",
    }

    news_dict = load_existing_news(company_name)
    response  = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  ❌ Erreur {response.status_code} pour {company_name} : "
              f"{response.json().get('message', '')}")
        return news_dict

    articles  = response.json().get("articles", [])
    new_count = 0

    for article in articles:
        title       = article.get("title", "")       or ""
        description = article.get("description", "") or ""
        source      = article.get("source", {}).get("name", "")
        published   = article.get("publishedAt", "")

        # Filtre de pertinence : le nom doit apparaître dans le titre ou la description
        search_name = get_search_name(company_name).lower()
        if search_name not in title.lower() and search_name not in description.lower():
            continue

        date = published.split("T")[0]
        if date not in news_dict:
            news_dict[date] = []

        # Déduplication par titre
        existing_titles = [a["title"] for a in news_dict[date]]
        if title not in existing_titles:
            news_dict[date].append({
                "title":       title,
                "description": description,
                "source":      source,
                "publishedAt": published,
            })
            new_count += 1

    # Sauvegarde JSON
    path = os.path.join(NEWS_FOLDER, f"{company_name.replace(' ', '_')}.json")
    with open(path, "w") as f:
        json.dump(news_dict, f, indent=4, ensure_ascii=False)

    total = sum(len(v) for v in news_dict.values())
    print(f"  ✅ {company_name:<30} {new_count:>3} nouveaux  |  {total:>3} total  →  {path}")
    return news_dict


# ── Lancement du scraping ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TP6 — Scraping de news financières")
    print("=" * 60)

    all_news = {}
    for i, company in enumerate(companies.keys()):
        all_news[company] = get_news_by_date(company)
        time.sleep(1)  # Respect du rate limit NewsAPI
        if (i + 1) % 20 == 0:
            print("  ⏸  Pause 30 secondes (rate limit)...")
            time.sleep(30)

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Résumé — Articles collectés par entreprise")
    print("=" * 60)
    for company, news in all_news.items():
        total = sum(len(v) for v in news.values())
        if total > 0:
            print(f"  {company:<35} {total:>3} articles")

    # ── Aperçu DataFrame ──────────────────────────────────────────────────────
    sample = "Microsoft"
    if sample in all_news and all_news[sample]:
        rows = []
        for date, articles in all_news[sample].items():
            for article in articles:
                rows.append({"date": date, **article})
        df = pd.DataFrame(rows)
        print(f"\nAperçu des news pour {sample} ({len(df)} articles) :")
        print(df[["date", "title", "source"]].head(5).to_string(index=False))
