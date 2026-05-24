# =============================================================================
# TP2 — Clustering et segmentation de l'univers d'investissement
# Automated Portfolio Management — Pratique de la Data Science 2024/2025
# =============================================================================
# Prérequis : pip install scikit-learn matplotlib scipy numpy pandas
# Inputs    : data/ratios_financiers.csv + Companies_historical_data/*.csv
# Outputs   : outputs/clustering_*.png
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

RATIOS_PATH = "data/ratios_financiers.csv"
HIST_FOLDER = "Companies_historical_data"
OUT_DIR     = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# MODULE A — Profils financiers (KMeans)
# =============================================================================
def preprocess_for_financial_clustering(filepath):
    """
    Charge les ratios financiers, sélectionne les 7 colonnes pertinentes,
    supprime les entreprises incomplètes (dropna) et standardise (Z-score).
    """
    df = pd.read_csv(filepath, index_col=0)
    cols = ["forwardPE", "beta", "priceToBook", "returnOnEquity",
            "returnOnAssets", "operatingMargins", "profitMargins"]
    existing = [c for c in cols if c in df.columns]
    df_clean = df[existing].dropna()
    scaler   = StandardScaler()
    return scaler.fit_transform(df_clean), df_clean


def elbow_method(data, max_k=12, save_path=None):
    """Méthode du coude : trace l'inertie KMeans pour K=1 à max_k."""
    inertias = []
    K_range  = range(1, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(data)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, "go-", label="Inertie")
    ax.set_title("Méthode du Coude")
    ax.set_xlabel("Nombre de clusters K")
    ax.set_ylabel("Inertie")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return inertias


def do_kmeans_clustering(data_scaled, df_original, n_clusters=5, save_path=None):
    """
    Applique KMeans (n_init=10), affiche les profils moyens par cluster
    et génère la visualisation t-SNE colorée par cluster.
    """
    km     = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(data_scaled)

    df_clust          = df_original.copy()
    df_clust["Cluster"] = labels

    print(f"\n--- Profils moyens (K={n_clusters}) ---")
    print(df_clust.groupby("Cluster").mean().to_string())

    # Visualisation t-SNE
    perp     = min(30, len(df_original) - 1)
    tsne     = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", learning_rate="auto")
    tsne_res = tsne.fit_transform(data_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(tsne_res[:, 0], tsne_res[:, 1],
                    c=labels, cmap="viridis", s=100, edgecolors="black")
    for i, name in enumerate(df_clust.index):
        ax.annotate(name, (tsne_res[i, 0], tsne_res[i, 1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.colorbar(sc, ax=ax, label="Cluster ID")
    ax.set_title(f"Visualisation t-SNE des profils financiers (K={n_clusters})")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return df_clust


# =============================================================================
# MODULE B — Profils de risque (Clustering Hiérarchique Ward)
# =============================================================================
def preprocess_for_risk_clustering(filepath):
    """
    Sélectionne les 5 variables de risque et standardise.
    """
    df       = pd.read_csv(filepath, index_col=0)
    cols     = ["beta", "debtToEquity", "currentRatio", "quickRatio", "operatingMargins"]
    existing = [c for c in cols if c in df.columns]
    df_risk  = df[existing].dropna()
    scaler   = StandardScaler()
    return scaler.fit_transform(df_risk), df_risk


def plot_dendrogram(data_scaled, df_risk, k_manuel=3, save_path=None):
    """
    Calcule le dendrogramme Ward et trace la coupe à k_manuel clusters.
    """
    linked    = linkage(data_scaled, method="ward")
    distances = linked[:, 2]
    threshold = (distances[-k_manuel] + distances[-k_manuel + 1]) / 2

    fig, ax = plt.subplots(figsize=(15, 7))
    dendrogram(linked, labels=df_risk.index,
               leaf_rotation=90, leaf_font_size=9,
               color_threshold=threshold, ax=ax)
    ax.axhline(y=threshold, color="r", linestyle="--",
               label=f"Coupe manuelle K={k_manuel}")
    ax.set_title(f"Analyse Hiérarchique — Profils de Risque (K={k_manuel})")
    ax.set_ylabel("Distance (Indice de dissimilarité)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Attribution finale des clusters
    hc      = AgglomerativeClustering(n_clusters=k_manuel, linkage="ward")
    labels  = hc.fit_predict(data_scaled)
    df_risk_final         = df_risk.copy()
    df_risk_final["Cluster"] = labels

    print(f"\n--- Profils moyens de risque (K={k_manuel}) ---")
    print(df_risk_final.groupby("Cluster").mean().to_string())

    return df_risk_final, linked


# =============================================================================
# MODULE C — Corrélations des rendements quotidiens
# =============================================================================
def preprocess_returns_data(folder_path):
    """
    Charge la colonne 'Rendement' de chaque CSV historique.
    Remplissage des NaN par la moyenne de la colonne (stable pour les corrélations).
    """
    returns_dict = {}
    for f in glob.glob(f"{folder_path}/*.csv"):
        name   = os.path.basename(f).split("_history")[0]
        df_tmp = pd.read_csv(f, index_col=0)
        if "Rendement" in df_tmp.columns:
            returns_dict[name] = df_tmp["Rendement"]

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.fillna(returns_df.mean())
    return returns_df


def do_correlation_clustering(returns_df, save_path=None):
    """
    Clustering hiérarchique Ward sur la matrice de corrélation des rendements.
    """
    corr   = returns_df.corr()
    linked = linkage(corr, method="ward")

    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linked, labels=corr.columns,
               leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Clustering par Corrélation des Rendements Quotidiens")
    ax.set_ylabel("Distance de Ward (Basée sur la Corrélation)")
    ax.axhline(y=1.5, color="r", linestyle="--")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Extrait quelques corrélations notables
    print("\n--- Extrait de la matrice de corrélation ---")
    print(corr.iloc[:5, :4].round(3).to_string())

    return corr, linked


# =============================================================================
# MODULE D — DBSCAN + Évaluation comparative (Silhouette Scores)
# =============================================================================
def find_best_dbscan(data_scaled):
    """Recherche automatique du premier eps produisant > 2 clusters."""
    for e in np.linspace(0.01, 3.0, 1000):
        labels    = DBSCAN(eps=e, min_samples=5).fit_predict(data_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 2:
            print(f"  ✅ eps={e:.2f} → {n_clusters} clusters trouvés")
            return labels
    print("  ❌ DBSCAN : aucune configuration valide trouvée")
    return np.zeros(len(data_scaled))


def do_dbscan_clustering(data_scaled, eps=1.2, min_samples=3):
    """Applique DBSCAN. Les points -1 sont des outliers (bruit)."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data_scaled)
    n_noise = np.sum(labels == -1)
    print(f"  DBSCAN : {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
          f"{n_noise} outliers")
    return labels


def evaluate_algorithms(data_scaled, n_clusters=3):
    """
    Compare KMeans, Hierarchical et DBSCAN via le Silhouette Score.
    Score de -1 (mauvais) à 1 (parfait). NaN si DBSCAN ne converge pas.
    """
    results = {}

    # KMeans
    km_labels        = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(data_scaled)
    results["K-Means"]      = silhouette_score(data_scaled, km_labels)

    # Hierarchical Ward
    hier_labels      = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data_scaled)
    results["Hierarchical"] = silhouette_score(data_scaled, hier_labels)

    # DBSCAN
    db_labels        = do_dbscan_clustering(data_scaled, eps=1.2, min_samples=3)
    mask             = db_labels != -1
    n_valid          = len(set(db_labels[mask]))
    if n_valid > 1 and np.sum(mask) > n_valid:
        results["DBSCAN"] = silhouette_score(data_scaled[mask], db_labels[mask])
    else:
        results["DBSCAN"] = np.nan
        print("  ⚠️  DBSCAN : pas assez de clusters valides pour le Silhouette Score")

    return results


# =============================================================================
# Lancement
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TP2 — Clustering et segmentation de l'univers")
    print("=" * 60)

    # ── Module A : Profils financiers ──────────────────────────────────────────
    print("\n[A] Profils financiers — KMeans")
    data_fin, df_fin = preprocess_for_financial_clustering(RATIOS_PATH)
    elbow_method(data_fin, save_path=f"{OUT_DIR}/clustering_elbow.png")
    print("  Figure sauvegardée → outputs/clustering_elbow.png")
    do_kmeans_clustering(data_fin, df_fin, n_clusters=5,
                         save_path=f"{OUT_DIR}/clustering_tsne_finance.png")
    print("  Figure sauvegardée → outputs/clustering_tsne_finance.png")

    # ── Module B : Profils de risque ───────────────────────────────────────────
    print("\n[B] Profils de risque — Hierarchical Ward")
    data_risk, df_risk = preprocess_for_risk_clustering(RATIOS_PATH)
    plot_dendrogram(data_risk, df_risk, k_manuel=3,
                    save_path=f"{OUT_DIR}/clustering_dendro_risk.png")
    print("  Figure sauvegardée → outputs/clustering_dendro_risk.png")

    # ── Module C : Corrélations des rendements ─────────────────────────────────
    print("\n[C] Corrélations des rendements quotidiens")
    returns_df = preprocess_returns_data(HIST_FOLDER)
    do_correlation_clustering(returns_df,
                              save_path=f"{OUT_DIR}/clustering_dendro_corr.png")
    print("  Figure sauvegardée → outputs/clustering_dendro_corr.png")

    # ── Module D : Tableau comparatif Silhouette ───────────────────────────────
    print("\n[D] Comparaison des algorithmes — Silhouette Scores")
    corr_matrix = returns_df.corr().values

    datasets = {
        "Financial Profile": data_fin,
        "Risk Profile":      data_risk,
        "Returns Corr":      corr_matrix,
    }
    comparison = {}
    for name, data in datasets.items():
        comparison[name] = evaluate_algorithms(data, n_clusters=3)

    df_eval = pd.DataFrame(comparison).T
    print("\n--- TABLEAU COMPARATIF DES SILHOUETTE SCORES ---")
    print(df_eval.round(4).to_string())

    print("\n" + "=" * 60)
    print("  ✅ TP2 terminé. Figures dans outputs/")
    print("=" * 60)
