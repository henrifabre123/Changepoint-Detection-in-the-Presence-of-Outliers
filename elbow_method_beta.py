"""
Elbow Method for Beta Tuning in Changepoint Detection

Ce fichier implémente une méthode systématique pour choisir le paramètre beta
en testant plusieurs multiples et en identifiant les changepoints stables.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm


# =============================================================================
# ELBOW METHOD - À copier dans votre notebook
# =============================================================================

def elbow_method_beta(
    y,
    loss: str = 'biweight',
    beta_multiples: list = None,
    K: float = None,
    tolerance: int = 50,  # Tolérance pour considérer deux CP comme "le même"
    # Fonctions à passer depuis le notebook
    rfpop_algorithm1_main=None,
    gamma_builder_biweight=None,
    gamma_builder_huber=None,
    gamma_builder_L2=None,
    compute_penalty_beta=None,
    compute_loss_bound_K=None,
):
    """
    Teste plusieurs valeurs de beta et identifie les changepoints stables.

    Parameters:
    -----------
    y : array-like
        Série temporelle
    loss : str
        Type de loss ('biweight', 'huber', 'l2')
    beta_multiples : list
        Liste des multiples de beta à tester (défaut: [0.5, 1, 2, 5, 10, 20, 50, 100])
    K : float
        Paramètre K (si None, calculé automatiquement)
    tolerance : int
        Deux changepoints à distance <= tolerance sont considérés comme identiques

    Returns:
    --------
    dict avec:
        - 'beta_values': les valeurs de beta testées
        - 'n_changepoints': nombre de CP pour chaque beta
        - 'changepoints_per_beta': liste des CP pour chaque beta
        - 'stable_changepoints': CP présents dans plusieurs valeurs de beta
        - 'stability_scores': score de stabilité pour chaque CP
        - 'elbow_beta': beta suggéré par la méthode du coude
    """
    if beta_multiples is None:
        beta_multiples = [0.5, 1, 2, 5, 10, 20, 50, 100]

    y_list = list(y) if not isinstance(y, list) else y
    y_arr = np.array(y_list)
    n = len(y_arr)

    # Calculer beta de base et K
    beta_paper = compute_penalty_beta(y_arr, loss)
    if K is None and loss in ['huber', 'biweight']:
        K = compute_loss_bound_K(y_arr, loss)

    print(f"Beta paper (base): {beta_paper:.4f}")
    print(f"K: {K:.4f}" if K else "K: N/A")
    print(f"Testing {len(beta_multiples)} beta values...")

    # Construire gamma_builder
    if loss == 'huber':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_huber(y_t, K, t)
    elif loss == 'biweight':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_biweight(y_t, K, t)
    else:
        gamma_builder = lambda y_t, t: gamma_builder_L2(y_t, t)

    results = {
        'beta_multiples': beta_multiples,
        'beta_values': [],
        'n_changepoints': [],
        'changepoints_per_beta': [],
        'beta_paper': beta_paper,
        'K': K,
        'n': n
    }

    # Tester chaque valeur de beta
    for mult in tqdm(beta_multiples, desc="Testing beta values"):
        beta = mult * beta_paper
        results['beta_values'].append(beta)

        # Exécuter l'algorithme
        cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_list, gamma_builder, beta)

        # Extraire les changepoints
        changepoints = extract_changepoints(cp_tau)

        results['n_changepoints'].append(len(changepoints))
        results['changepoints_per_beta'].append(changepoints)

    # Analyser la stabilité des changepoints
    stability_analysis = analyze_stability(
        results['changepoints_per_beta'],
        results['beta_multiples'],
        tolerance=tolerance,
        n=n
    )
    results.update(stability_analysis)

    # Trouver le coude
    elbow_idx = find_elbow(results['beta_multiples'], results['n_changepoints'])
    results['elbow_idx'] = elbow_idx
    results['elbow_beta_multiple'] = beta_multiples[elbow_idx]
    results['elbow_beta'] = beta_multiples[elbow_idx] * beta_paper

    return results


def extract_changepoints(cp_tau):
    """Extrait les changepoints par backtracking."""
    n = len(cp_tau)
    changepoints = []
    idx = n - 1

    while idx > 0:
        tau = cp_tau[idx]
        if tau > 0:
            changepoints.append(tau)
        idx = tau

    return sorted([cp for cp in changepoints if cp > 0])


def analyze_stability(changepoints_per_beta, beta_multiples, tolerance, n):
    """
    Analyse la stabilité des changepoints à travers différentes valeurs de beta.

    Un changepoint est "stable" s'il apparaît (à ±tolerance près) dans plusieurs
    valeurs de beta.
    """
    # Collecter tous les changepoints uniques (avec clustering par tolérance)
    all_cps = []
    for cps in changepoints_per_beta:
        all_cps.extend(cps)

    if not all_cps:
        return {
            'stable_changepoints': [],
            'stability_scores': {},
            'cp_presence_matrix': []
        }

    # Clustering des changepoints proches
    all_cps_sorted = sorted(set(all_cps))
    clusters = []
    current_cluster = [all_cps_sorted[0]]

    for cp in all_cps_sorted[1:]:
        if cp - current_cluster[-1] <= tolerance:
            current_cluster.append(cp)
        else:
            clusters.append(current_cluster)
            current_cluster = [cp]
    clusters.append(current_cluster)

    # Représentant de chaque cluster (médiane)
    cluster_representatives = [int(np.median(c)) for c in clusters]

    # Matrice de présence: pour chaque cluster, dans combien de beta est-il présent?
    presence_matrix = []
    for i, cps in enumerate(changepoints_per_beta):
        row = []
        for cluster in clusters:
            # Le cluster est-il présent dans cette configuration?
            present = any(any(abs(cp - c) <= tolerance for c in cluster) for cp in cps)
            row.append(1 if present else 0)
        presence_matrix.append(row)

    presence_matrix = np.array(presence_matrix)

    # Score de stabilité: proportion de betas où le CP est présent
    stability_scores = {}
    for i, rep in enumerate(cluster_representatives):
        score = presence_matrix[:, i].sum() / len(beta_multiples)
        stability_scores[rep] = score

    # Changepoints stables (présents dans >= 50% des configurations)
    stable_cps = [cp for cp, score in stability_scores.items() if score >= 0.5]

    return {
        'stable_changepoints': sorted(stable_cps),
        'stability_scores': stability_scores,
        'cp_clusters': clusters,
        'cluster_representatives': cluster_representatives,
        'presence_matrix': presence_matrix
    }


def find_elbow(x_values, y_values):
    """
    Trouve le point de coude en utilisant la méthode de la distance maximale
    à la ligne reliant le premier et dernier point.
    """
    x = np.array(x_values)
    y = np.array(y_values)

    # Normaliser
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Ligne entre premier et dernier point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    # Distance de chaque point à la ligne
    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        # Distance point-ligne
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(d)

    return np.argmax(distances)


def plot_elbow_analysis(results, title="", figsize=(16, 12)):
    """
    Visualisation complète de l'analyse elbow.

    4 subplots:
    1. Elbow plot (n_changepoints vs beta)
    2. Positions des changepoints pour chaque beta
    3. Heatmap de stabilité
    4. Changepoints stables avec leurs scores
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Elbow plot
    ax1 = axes[0, 0]
    ax1.plot(results['beta_multiples'], results['n_changepoints'], 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=results['elbow_beta_multiple'], color='red', linestyle='--',
                label=f'Elbow: {results["elbow_beta_multiple"]}x')
    ax1.set_xscale('log')
    ax1.set_xlabel('Beta multiple', fontsize=12)
    ax1.set_ylabel('Number of changepoints', fontsize=12)
    ax1.set_title('Elbow Plot', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annoter les points
    for i, (x, y) in enumerate(zip(results['beta_multiples'], results['n_changepoints'])):
        ax1.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    # 2. Positions des changepoints pour chaque beta
    ax2 = axes[0, 1]
    for i, (mult, cps) in enumerate(zip(results['beta_multiples'], results['changepoints_per_beta'])):
        y_pos = [i] * len(cps)
        ax2.scatter(cps, y_pos, s=50, alpha=0.7)

    ax2.set_yticks(range(len(results['beta_multiples'])))
    ax2.set_yticklabels([f'{m}x' for m in results['beta_multiples']])
    ax2.set_xlabel('Changepoint position', fontsize=12)
    ax2.set_ylabel('Beta multiple', fontsize=12)
    ax2.set_title('Changepoint Positions per Beta', fontsize=14)
    ax2.set_xlim(0, results['n'])
    ax2.grid(True, alpha=0.3, axis='x')

    # Marquer les changepoints stables
    if results['stable_changepoints']:
        for cp in results['stable_changepoints']:
            ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 3. Heatmap de stabilité
    ax3 = axes[1, 0]
    if len(results.get('presence_matrix', [])) > 0 and len(results.get('cluster_representatives', [])) > 0:
        im = ax3.imshow(results['presence_matrix'], aspect='auto', cmap='Blues')
        ax3.set_yticks(range(len(results['beta_multiples'])))
        ax3.set_yticklabels([f'{m}x' for m in results['beta_multiples']])
        ax3.set_xticks(range(len(results['cluster_representatives'])))
        ax3.set_xticklabels(results['cluster_representatives'], rotation=45, ha='right')
        ax3.set_xlabel('Changepoint cluster (position)', fontsize=12)
        ax3.set_ylabel('Beta multiple', fontsize=12)
        ax3.set_title('Stability Heatmap (presence across beta values)', fontsize=14)
        plt.colorbar(im, ax=ax3, label='Present')
    else:
        ax3.text(0.5, 0.5, 'No changepoints detected', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Stability Heatmap', fontsize=14)

    # 4. Scores de stabilité
    ax4 = axes[1, 1]
    if results['stability_scores']:
        cps = sorted(results['stability_scores'].keys())
        scores = [results['stability_scores'][cp] for cp in cps]
        colors = ['green' if s >= 0.5 else 'gray' for s in scores]
        bars = ax4.bar(range(len(cps)), scores, color=colors, alpha=0.7)
        ax4.axhline(y=0.5, color='red', linestyle='--', label='Stability threshold (50%)')
        ax4.set_xticks(range(len(cps)))
        ax4.set_xticklabels(cps, rotation=45, ha='right')
        ax4.set_xlabel('Changepoint position', fontsize=12)
        ax4.set_ylabel('Stability score', fontsize=12)
        ax4.set_title('Changepoint Stability Scores', fontsize=14)
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No changepoints detected', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Changepoint Stability Scores', fontsize=14)

    plt.suptitle(f'{title}\nElbow Method Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def print_summary(results):
    """Affiche un résumé des résultats."""
    print("\n" + "="*60)
    print("ELBOW METHOD SUMMARY")
    print("="*60)

    print(f"\nBeta paper (base): {results['beta_paper']:.4f}")
    print(f"Data length: {results['n']}")

    print(f"\n--- Results per Beta ---")
    for mult, n_cp, cps in zip(results['beta_multiples'],
                               results['n_changepoints'],
                               results['changepoints_per_beta']):
        print(f"  {mult:6.1f}x beta: {n_cp:3d} changepoints")

    print(f"\n--- Elbow Point ---")
    print(f"  Suggested beta multiple: {results['elbow_beta_multiple']}x")
    print(f"  Suggested beta value: {results['elbow_beta']:.4f}")

    print(f"\n--- Stable Changepoints (>= 50% presence) ---")
    if results['stable_changepoints']:
        for cp in results['stable_changepoints']:
            score = results['stability_scores'][cp]
            print(f"  Position {cp:5d}: stability = {score:.1%}")
    else:
        print("  No stable changepoints found")

    print("="*60)


# =============================================================================
# CODE COMPLET À COPIER DANS VOTRE NOTEBOOK
# =============================================================================

NOTEBOOK_CODE = '''
# =============================================================================
# ELBOW METHOD POUR BETA - À copier dans votre notebook
# =============================================================================

def elbow_method_beta(
    y,
    loss: str = 'biweight',
    beta_multiples: list = None,
    K: float = None,
    tolerance: int = 50,
):
    """
    Teste plusieurs valeurs de beta et identifie les changepoints stables.
    """
    if beta_multiples is None:
        beta_multiples = [0.5, 1, 2, 5, 10, 20, 50, 100]

    y_list = list(y) if not isinstance(y, list) else y
    y_arr = np.array(y_list)
    n = len(y_arr)

    beta_paper = compute_penalty_beta(y_arr, loss)
    if K is None and loss in ['huber', 'biweight']:
        K = compute_loss_bound_K(y_arr, loss)

    print(f"Beta paper (base): {beta_paper:.4f}")
    print(f"K: {K:.4f}" if K else "K: N/A")

    if loss == 'huber':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_huber(y_t, K, t)
    elif loss == 'biweight':
        gamma_builder = lambda y_t, t, K=K: gamma_builder_biweight(y_t, K, t)
    else:
        gamma_builder = lambda y_t, t: gamma_builder_L2(y_t, t)

    results = {
        'beta_multiples': beta_multiples,
        'beta_values': [],
        'n_changepoints': [],
        'changepoints_per_beta': [],
        'beta_paper': beta_paper,
        'K': K,
        'n': n
    }

    for mult in tqdm(beta_multiples, desc="Testing beta values"):
        beta = mult * beta_paper
        results['beta_values'].append(beta)

        cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_list, gamma_builder, beta)

        # Extraction
        changepoints = []
        idx = n - 1
        while idx > 0:
            tau = cp_tau[idx]
            if tau > 0:
                changepoints.append(tau)
            idx = tau
        changepoints = sorted([cp for cp in changepoints if cp > 0])

        results['n_changepoints'].append(len(changepoints))
        results['changepoints_per_beta'].append(changepoints)

    # Analyse de stabilité
    results.update(analyze_cp_stability(
        results['changepoints_per_beta'],
        results['beta_multiples'],
        tolerance=tolerance,
        n=n
    ))

    # Trouver le coude
    elbow_idx = find_elbow_point(results['beta_multiples'], results['n_changepoints'])
    results['elbow_idx'] = elbow_idx
    results['elbow_beta_multiple'] = beta_multiples[elbow_idx]
    results['elbow_beta'] = beta_multiples[elbow_idx] * beta_paper

    return results


def analyze_cp_stability(changepoints_per_beta, beta_multiples, tolerance, n):
    """Analyse la stabilité des changepoints."""
    all_cps = []
    for cps in changepoints_per_beta:
        all_cps.extend(cps)

    if not all_cps:
        return {
            'stable_changepoints': [],
            'stability_scores': {},
            'presence_matrix': np.array([]),
            'cluster_representatives': []
        }

    # Clustering
    all_cps_sorted = sorted(set(all_cps))
    clusters = []
    current_cluster = [all_cps_sorted[0]]

    for cp in all_cps_sorted[1:]:
        if cp - current_cluster[-1] <= tolerance:
            current_cluster.append(cp)
        else:
            clusters.append(current_cluster)
            current_cluster = [cp]
    clusters.append(current_cluster)

    cluster_representatives = [int(np.median(c)) for c in clusters]

    # Matrice de présence
    presence_matrix = []
    for cps in changepoints_per_beta:
        row = []
        for cluster in clusters:
            present = any(any(abs(cp - c) <= tolerance for c in cluster) for cp in cps)
            row.append(1 if present else 0)
        presence_matrix.append(row)

    presence_matrix = np.array(presence_matrix)

    # Scores de stabilité
    stability_scores = {}
    for i, rep in enumerate(cluster_representatives):
        score = presence_matrix[:, i].sum() / len(beta_multiples)
        stability_scores[rep] = score

    stable_cps = [cp for cp, score in stability_scores.items() if score >= 0.5]

    return {
        'stable_changepoints': sorted(stable_cps),
        'stability_scores': stability_scores,
        'cp_clusters': clusters,
        'cluster_representatives': cluster_representatives,
        'presence_matrix': presence_matrix
    }


def find_elbow_point(x_values, y_values):
    """Trouve le coude par distance max à la ligne."""
    x = np.array(x_values)
    y = np.array(y_values)

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(d)

    return np.argmax(distances)


def plot_elbow_analysis(results, title=""):
    """Visualisation complète."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Elbow plot
    ax1 = axes[0, 0]
    ax1.plot(results['beta_multiples'], results['n_changepoints'], 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=results['elbow_beta_multiple'], color='red', linestyle='--',
                label=f'Elbow: {results["elbow_beta_multiple"]}x')
    ax1.set_xscale('log')
    ax1.set_xlabel('Beta multiple')
    ax1.set_ylabel('Number of changepoints')
    ax1.set_title('Elbow Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(results['beta_multiples'], results['n_changepoints'])):
        ax1.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    # 2. Positions des CP
    ax2 = axes[0, 1]
    for i, (mult, cps) in enumerate(zip(results['beta_multiples'], results['changepoints_per_beta'])):
        ax2.scatter(cps, [i] * len(cps), s=50, alpha=0.7)
    ax2.set_yticks(range(len(results['beta_multiples'])))
    ax2.set_yticklabels([f'{m}x' for m in results['beta_multiples']])
    ax2.set_xlabel('Changepoint position')
    ax2.set_ylabel('Beta multiple')
    ax2.set_title('Changepoint Positions per Beta')
    ax2.set_xlim(0, results['n'])
    for cp in results['stable_changepoints']:
        ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Heatmap
    ax3 = axes[1, 0]
    if len(results['presence_matrix']) > 0:
        im = ax3.imshow(results['presence_matrix'], aspect='auto', cmap='Blues')
        ax3.set_yticks(range(len(results['beta_multiples'])))
        ax3.set_yticklabels([f'{m}x' for m in results['beta_multiples']])
        ax3.set_xticks(range(len(results['cluster_representatives'])))
        ax3.set_xticklabels(results['cluster_representatives'], rotation=45, ha='right')
        ax3.set_xlabel('Changepoint cluster')
        ax3.set_ylabel('Beta multiple')
        ax3.set_title('Stability Heatmap')
        plt.colorbar(im, ax=ax3)

    # 4. Scores
    ax4 = axes[1, 1]
    if results['stability_scores']:
        cps = sorted(results['stability_scores'].keys())
        scores = [results['stability_scores'][cp] for cp in cps]
        colors = ['green' if s >= 0.5 else 'gray' for s in scores]
        ax4.bar(range(len(cps)), scores, color=colors, alpha=0.7)
        ax4.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
        ax4.set_xticks(range(len(cps)))
        ax4.set_xticklabels(cps, rotation=45, ha='right')
        ax4.set_xlabel('Changepoint position')
        ax4.set_ylabel('Stability score')
        ax4.set_title('Stability Scores')
        ax4.set_ylim(0, 1.1)
        ax4.legend()

    plt.suptitle(f'{title}\\nElbow Method Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# UTILISATION
# =============================================================================

col = 'T10Y2Y'
y = df[col].dropna()
y = y[y.index > '2000']

print(f"Série: {col}, n = {len(y)}")

# Lancer l'analyse
results = elbow_method_beta(
    y.values,
    loss='biweight',
    beta_multiples=[0.5, 1, 2, 5, 10, 20, 50, 100],
    tolerance=50  # CP à ±50 points sont considérés comme identiques
)

# Visualisation
plot_elbow_analysis(results, title=col)

# Résumé
print("\\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Suggested beta multiple: {results['elbow_beta_multiple']}x")
print(f"Stable changepoints: {results['stable_changepoints']}")
for cp in results['stable_changepoints']:
    print(f"  Position {cp}: stability = {results['stability_scores'][cp]:.1%}")
'''


if __name__ == "__main__":
    print("Elbow Method for Beta Tuning")
    print("\nCopy the code from NOTEBOOK_CODE into your notebook.")
    print("\nOr import the functions:")
    print("  from elbow_method_beta import elbow_method_beta, plot_elbow_analysis")
