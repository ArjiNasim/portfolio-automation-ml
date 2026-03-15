%%writefile 2_clustering.py
"""
TP2 - Clustering Analysis

This script applies several clustering algorithms to the datasets created in TP1.

We study three different clustering problems:
1) Financial profiles clustering
2) Risk profiles clustering
3) Daily returns correlation clustering
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1) Paths and folders
# ============================================================

BASE_DIR = Path(".")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
HISTORICAL_DATA_DIR = RAW_DATA_DIR / "companies_historical_data"

OUTPUTS_DIR = BASE_DIR / "outputs" / "clustering"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) Feature selection
# ============================================================

FINANCIAL_FEATURES = [
    "forwardPE",
    "priceToBook",
    "priceToSales",
    "trailingEps",
    "returnOnEquity",
    "returnOnAssets",
    "operatingMargins",
    "profitMargins",
]

RISK_FEATURES = [
    "beta",
    "debtToEquity",
    "currentRatio",
    "quickRatio",
    "returnOnEquity",
    "returnOnAssets",
]


# ============================================================
# 3) Data loading functions
# ============================================================

def load_financial_ratios(file_path: Path = RAW_DATA_DIR / "financial_ratios.csv") -> pd.DataFrame:
    """
    Load the financial ratios dataset created in TP1.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, index_col=0)
    return df


def find_return_column(df: pd.DataFrame) -> str:
    """
    Detect the name of the return column.
    """
    possible_columns = ["Return", "Daily Return", "Rendement"]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(f"No return column found. Expected one of: {possible_columns}")


def load_returns_dataframe(folder_path: Path = HISTORICAL_DATA_DIR) -> pd.DataFrame:
    """
    Load all company historical CSV files and build one dataframe
    containing daily returns for all companies.
    """
    file_paths = glob.glob(str(folder_path / "*.csv"))

    if len(file_paths) == 0:
        raise FileNotFoundError(f"No historical CSV files found in {folder_path}")

    returns_dict = {}

    for file_path in file_paths:
        file_path = Path(file_path)

        company_name = file_path.stem.replace("_historical_data", "").replace("_", " ")

        df = pd.read_csv(file_path)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        return_col = find_return_column(df)
        returns_dict[company_name] = df[return_col]

    returns_df = pd.DataFrame(returns_dict)

    # Fill missing returns with the median of each column
    returns_df = returns_df.apply(lambda col: col.fillna(col.median()), axis=0)

    # Remove completely empty columns if any
    returns_df = returns_df.dropna(axis=1, how="all")

    if returns_df.empty:
        raise ValueError("The returns dataframe is empty after loading and cleaning.")

    print("Returns dataframe shape:", returns_df.shape)

    return returns_df


# ============================================================
# 4) Preprocessing functions
# ============================================================

def preprocess_dataset(
    df: pd.DataFrame,
    selected_features: List[str],
    standardize: bool = True,
    max_missing_ratio: float = 0.80
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select relevant variables, handle missing values properly,
    and standardize the data.
    """
    working_df = df[selected_features].copy()

    print("\nInitial dataset shape:", working_df.shape)

    # Missing ratio by feature
    missing_ratio = working_df.isna().mean()

    print("Missing value ratio per feature:")
    print(missing_ratio)

    # Keep columns with acceptable missing ratio
    kept_columns = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    working_df = working_df[kept_columns]

    print("Features kept:", kept_columns)

    if working_df.shape[1] == 0:
        raise ValueError("No feature left after removing columns with too many missing values.")

    # Fill remaining missing values with median
    for col in working_df.columns:
        col_median = working_df[col].median()

        # If the whole column is NaN, replace with 0
        if pd.isna(col_median):
            col_median = 0.0

        working_df[col] = working_df[col].fillna(col_median)

    # Remove remaining problematic rows if any
    working_df = working_df.dropna(axis=0)

    print("Final dataset shape after cleaning:", working_df.shape)

    if working_df.empty:
        raise ValueError("The preprocessed dataframe is empty after cleaning.")

    data = working_df.values

    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return working_df, data


def preprocess_correlation_matrix(returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build the correlation matrix from the returns dataframe.
    """
    corr_matrix = returns_df.corr()

    corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr_matrix = corr_matrix.fillna(0)

    scaler = StandardScaler()
    scaled_corr = scaler.fit_transform(corr_matrix)

    print("\nCorrelation matrix shape:", corr_matrix.shape)

    return corr_matrix, scaled_corr


# ============================================================
# 5) Evaluation helpers
# ============================================================

def safe_silhouette_score(data: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute silhouette score only when it is meaningful.
    """
    unique_labels = set(labels)

    if -1 in unique_labels:
        unique_labels = unique_labels - {-1}

    if len(unique_labels) < 2:
        return None

    try:
        return float(silhouette_score(data, labels))
    except Exception:
        return None


def describe_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """
    Return the number of observations per cluster.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def format_silhouette(value: Optional[float]) -> Optional[float]:
    """
    Round silhouette score only if it exists.
    """
    if value is None:
        return None
    return round(value, 4)


# ============================================================
# 6) Visualization functions
# ============================================================

def plot_elbow_curve(
    data: np.ndarray,
    dataset_name: str,
    k_range: range = range(2, 11),
    save_path: Optional[Path] = None
) -> None:
    """
    Plot elbow curve for KMeans.
    """
    inertias = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        model.fit(data)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title(f"Elbow Method - {dataset_name}")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_pca_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    index_labels: List[str],
    title: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Project data onto 2 principal components for visualization.
    """
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(data)

    plt.figure(figsize=(10, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=labels)

    for i, company in enumerate(index_labels):
        plt.annotate(company, (projected[i, 0], projected[i, 1]), fontsize=8, alpha=0.8)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_tsne_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    index_labels: List[str],
    title: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Use t-SNE for a non-linear 2D visualization.
    """
    n_samples = data.shape[0]
    perplexity = min(10, max(2, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    projected = tsne.fit_transform(data)

    plt.figure(figsize=(10, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=labels)

    for i, company in enumerate(index_labels):
        plt.annotate(company, (projected[i, 0], projected[i, 1]), fontsize=8, alpha=0.8)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_hierarchical_dendrogram(
    data: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot hierarchical clustering dendrogram.
    """
    linked = linkage(data, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.xlabel("Companies")
    plt.ylabel("Distance")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ============================================================
# 7) Clustering functions
# ============================================================

def apply_kmeans(data: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    return model.fit_predict(data)


def apply_hierarchical_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(data)


def apply_dbscan(data: np.ndarray, eps: float = 1.2, min_samples: int = 3) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(data)


# ============================================================
# 8) Cluster interpretation helpers
# ============================================================

def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute cluster-wise averages for interpretation.
    """
    temp_df = df.copy()
    temp_df["cluster"] = labels
    summary = temp_df.groupby("cluster").mean(numeric_only=True)
    return summary


def save_cluster_summary(summary_df: pd.DataFrame, file_name: str) -> None:
    output_path = TABLES_DIR / file_name
    summary_df.to_csv(output_path)


# ============================================================
# 9) Dataset-specific analyses
# ============================================================

def run_financial_profiles_analysis(ratios_df: pd.DataFrame) -> List[Dict]:
    """
    Financial profiles clustering:
    main algorithm = KMeans
    """
    print("\n" + "=" * 70)
    print("FINANCIAL PROFILES CLUSTERING")
    print("=" * 70)

    working_df, data = preprocess_dataset(
        ratios_df,
        selected_features=FINANCIAL_FEATURES,
        standardize=True,
        max_missing_ratio=0.80,
    )

    plot_elbow_curve(
        data,
        dataset_name="Financial Profiles",
        save_path=FIGURES_DIR / "financial_profiles_elbow.png",
    )

    n_clusters = 4
    results = []

    # KMeans
    kmeans_labels = apply_kmeans(data, n_clusters=n_clusters)
    kmeans_score = safe_silhouette_score(data, kmeans_labels)

    kmeans_summary = summarize_clusters(working_df, kmeans_labels)
    save_cluster_summary(kmeans_summary, "financial_profiles_kmeans_summary.csv")

    plot_pca_clusters(
        data,
        kmeans_labels,
        working_df.index.tolist(),
        title="Financial Profiles - KMeans (PCA)",
        save_path=FIGURES_DIR / "financial_profiles_kmeans_pca.png",
    )

    plot_tsne_clusters(
        data,
        kmeans_labels,
        working_df.index.tolist(),
        title="Financial Profiles - KMeans (t-SNE)",
        save_path=FIGURES_DIR / "financial_profiles_kmeans_tsne.png",
    )

    results.append({
        "dataset": "financial_profiles",
        "algorithm": "kmeans",
        "n_clusters_detected": len(np.unique(kmeans_labels)),
        "silhouette_score": format_silhouette(kmeans_score),
        "cluster_sizes": describe_cluster_sizes(kmeans_labels),
    })

    # Hierarchical
    hier_labels = apply_hierarchical_clustering(data, n_clusters=n_clusters)
    hier_score = safe_silhouette_score(data, hier_labels)

    hier_summary = summarize_clusters(working_df, hier_labels)
    save_cluster_summary(hier_summary, "financial_profiles_hierarchical_summary.csv")

    plot_hierarchical_dendrogram(
        data,
        working_df.index.tolist(),
        title="Financial Profiles - Hierarchical Dendrogram",
        save_path=FIGURES_DIR / "financial_profiles_hierarchical_dendrogram.png",
    )

    plot_pca_clusters(
        data,
        hier_labels,
        working_df.index.tolist(),
        title="Financial Profiles - Hierarchical Clustering (PCA)",
        save_path=FIGURES_DIR / "financial_profiles_hierarchical_pca.png",
    )

    results.append({
        "dataset": "financial_profiles",
        "algorithm": "hierarchical",
        "n_clusters_detected": len(np.unique(hier_labels)),
        "silhouette_score": format_silhouette(hier_score),
        "cluster_sizes": describe_cluster_sizes(hier_labels),
    })

    # DBSCAN
    dbscan_labels = apply_dbscan(data, eps=1.5, min_samples=3)
    dbscan_score = safe_silhouette_score(data, dbscan_labels)

    dbscan_summary = summarize_clusters(working_df, dbscan_labels)
    save_cluster_summary(dbscan_summary, "financial_profiles_dbscan_summary.csv")

    plot_pca_clusters(
        data,
        dbscan_labels,
        working_df.index.tolist(),
        title="Financial Profiles - DBSCAN (PCA)",
        save_path=FIGURES_DIR / "financial_profiles_dbscan_pca.png",
    )

    results.append({
        "dataset": "financial_profiles",
        "algorithm": "dbscan",
        "n_clusters_detected": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        "silhouette_score": format_silhouette(dbscan_score),
        "cluster_sizes": describe_cluster_sizes(dbscan_labels),
    })

    return results


def run_risk_profiles_analysis(ratios_df: pd.DataFrame) -> List[Dict]:
    """
    Risk profiles clustering:
    main algorithm = hierarchical clustering
    """
    print("\n" + "=" * 70)
    print("RISK PROFILES CLUSTERING")
    print("=" * 70)

    working_df, data = preprocess_dataset(
        ratios_df,
        selected_features=RISK_FEATURES,
        standardize=True,
        max_missing_ratio=0.80,
    )

    results = []
    n_clusters = 4

    # Hierarchical
    hier_labels = apply_hierarchical_clustering(data, n_clusters=n_clusters)
    hier_score = safe_silhouette_score(data, hier_labels)

    hier_summary = summarize_clusters(working_df, hier_labels)
    save_cluster_summary(hier_summary, "risk_profiles_hierarchical_summary.csv")

    plot_hierarchical_dendrogram(
        data,
        working_df.index.tolist(),
        title="Risk Profiles - Hierarchical Dendrogram",
        save_path=FIGURES_DIR / "risk_profiles_hierarchical_dendrogram.png",
    )

    plot_pca_clusters(
        data,
        hier_labels,
        working_df.index.tolist(),
        title="Risk Profiles - Hierarchical Clustering (PCA)",
        save_path=FIGURES_DIR / "risk_profiles_hierarchical_pca.png",
    )

    results.append({
        "dataset": "risk_profiles",
        "algorithm": "hierarchical",
        "n_clusters_detected": len(np.unique(hier_labels)),
        "silhouette_score": format_silhouette(hier_score),
        "cluster_sizes": describe_cluster_sizes(hier_labels),
    })

    # KMeans
    kmeans_labels = apply_kmeans(data, n_clusters=n_clusters)
    kmeans_score = safe_silhouette_score(data, kmeans_labels)

    kmeans_summary = summarize_clusters(working_df, kmeans_labels)
    save_cluster_summary(kmeans_summary, "risk_profiles_kmeans_summary.csv")

    plot_pca_clusters(
        data,
        kmeans_labels,
        working_df.index.tolist(),
        title="Risk Profiles - KMeans (PCA)",
        save_path=FIGURES_DIR / "risk_profiles_kmeans_pca.png",
    )

    results.append({
        "dataset": "risk_profiles",
        "algorithm": "kmeans",
        "n_clusters_detected": len(np.unique(kmeans_labels)),
        "silhouette_score": format_silhouette(kmeans_score),
        "cluster_sizes": describe_cluster_sizes(kmeans_labels),
    })

    # DBSCAN
    dbscan_labels = apply_dbscan(data, eps=1.4, min_samples=3)
    dbscan_score = safe_silhouette_score(data, dbscan_labels)

    dbscan_summary = summarize_clusters(working_df, dbscan_labels)
    save_cluster_summary(dbscan_summary, "risk_profiles_dbscan_summary.csv")

    plot_pca_clusters(
        data,
        dbscan_labels,
        working_df.index.tolist(),
        title="Risk Profiles - DBSCAN (PCA)",
        save_path=FIGURES_DIR / "risk_profiles_dbscan_pca.png",
    )

    results.append({
        "dataset": "risk_profiles",
        "algorithm": "dbscan",
        "n_clusters_detected": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        "silhouette_score": format_silhouette(dbscan_score),
        "cluster_sizes": describe_cluster_sizes(dbscan_labels),
    })

    return results


def run_returns_correlation_analysis(returns_df: pd.DataFrame) -> List[Dict]:
    """
    Daily returns correlation clustering:
    main algorithm = hierarchical clustering
    """
    print("\n" + "=" * 70)
    print("DAILY RETURNS CORRELATION CLUSTERING")
    print("=" * 70)

    corr_matrix, data = preprocess_correlation_matrix(returns_df)

    labels_list = corr_matrix.index.tolist()
    results = []
    n_clusters = 4

    # Hierarchical
    hier_labels = apply_hierarchical_clustering(data, n_clusters=n_clusters)
    hier_score = safe_silhouette_score(data, hier_labels)

    plot_hierarchical_dendrogram(
        data,
        labels_list,
        title="Daily Returns Correlations - Hierarchical Dendrogram",
        save_path=FIGURES_DIR / "returns_correlation_hierarchical_dendrogram.png",
    )

    plot_pca_clusters(
        data,
        hier_labels,
        labels_list,
        title="Daily Returns Correlations - Hierarchical Clustering (PCA)",
        save_path=FIGURES_DIR / "returns_correlation_hierarchical_pca.png",
    )

    hier_summary = summarize_clusters(corr_matrix, hier_labels)
    save_cluster_summary(hier_summary, "returns_correlation_hierarchical_summary.csv")

    results.append({
        "dataset": "returns_correlation",
        "algorithm": "hierarchical",
        "n_clusters_detected": len(np.unique(hier_labels)),
        "silhouette_score": format_silhouette(hier_score),
        "cluster_sizes": describe_cluster_sizes(hier_labels),
    })

    # KMeans
    kmeans_labels = apply_kmeans(data, n_clusters=n_clusters)
    kmeans_score = safe_silhouette_score(data, kmeans_labels)

    kmeans_summary = summarize_clusters(corr_matrix, kmeans_labels)
    save_cluster_summary(kmeans_summary, "returns_correlation_kmeans_summary.csv")

    plot_pca_clusters(
        data,
        kmeans_labels,
        labels_list,
        title="Daily Returns Correlations - KMeans (PCA)",
        save_path=FIGURES_DIR / "returns_correlation_kmeans_pca.png",
    )

    results.append({
        "dataset": "returns_correlation",
        "algorithm": "kmeans",
        "n_clusters_detected": len(np.unique(kmeans_labels)),
        "silhouette_score": format_silhouette(kmeans_score),
        "cluster_sizes": describe_cluster_sizes(kmeans_labels),
    })

    # DBSCAN
    dbscan_labels = apply_dbscan(data, eps=1.5, min_samples=3)
    dbscan_score = safe_silhouette_score(data, dbscan_labels)

    dbscan_summary = summarize_clusters(corr_matrix, dbscan_labels)
    save_cluster_summary(dbscan_summary, "returns_correlation_dbscan_summary.csv")

    plot_pca_clusters(
        data,
        dbscan_labels,
        labels_list,
        title="Daily Returns Correlations - DBSCAN (PCA)",
        save_path=FIGURES_DIR / "returns_correlation_dbscan_pca.png",
    )

    results.append({
        "dataset": "returns_correlation",
        "algorithm": "dbscan",
        "n_clusters_detected": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        "silhouette_score": format_silhouette(dbscan_score),
        "cluster_sizes": describe_cluster_sizes(dbscan_labels),
    })

    return results


# ============================================================
# 10) Global comparison table
# ============================================================

def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Convert all clustering results into one comparison dataframe.
    """
    comparison_df = pd.DataFrame(results)
    return comparison_df


# ============================================================
# 11) Main function
# ============================================================

def main() -> None:
    """
    Run the full TP2 clustering pipeline.
    """
    print("Loading datasets...")

    ratios_df = load_financial_ratios()
    returns_df = load_returns_dataframe()

    all_results = []

    all_results.extend(run_financial_profiles_analysis(ratios_df))
    all_results.extend(run_risk_profiles_analysis(ratios_df))
    all_results.extend(run_returns_correlation_analysis(returns_df))

    comparison_df = build_comparison_table(all_results)

    comparison_path = TABLES_DIR / "clustering_comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df)

    print(f"\nComparison table saved to: {comparison_path}")
    print("TP2 completed successfully.")


# ============================================================
# 12) Script entry point
# ============================================================

if __name__ == "__main__":
    main()
