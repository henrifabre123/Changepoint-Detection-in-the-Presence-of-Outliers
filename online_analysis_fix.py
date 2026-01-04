"""
Online Analysis Fix for Changepoint Detection

Ce fichier contient une version améliorée de l'analyse online qui corrige
les problèmes de chutes à 0 dans l'estimation du changepoint le plus récent.

Problèmes corrigés:
1. Beta trop élevé causant des sous-détections
2. K calculé sur données complètes au lieu de partielles
3. Absence de mécanisme de persistance des changepoints
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.robust import mad


def online_most_recent_changepoint_fixed(
    y,
    loss: str = 'biweight',
    beta: float = None,
    beta_scaling: float = 4.0,  # Réduit de 12 à 4
    K: float = None,
    recompute_K: bool = True,  # Recalculer K à chaque étape
    step: int = 50,
    min_obs: int = 100,
    # Importer les fonctions nécessaires depuis votre notebook
    rfpop_algorithm1_main=None,
    gamma_builder_biweight=None,
    gamma_builder_huber=None,
    gamma_builder_L2=None,
    compute_penalty_beta_func=None,
    compute_loss_bound_K_func=None,
):
    """
    Analyse online améliorée avec corrections pour éviter les chutes à 0.

    Changements par rapport à l'original:
    1. beta_scaling réduit (défaut: 4 au lieu de 12)
    2. Option recompute_K pour recalculer K sur les données partielles
    3. Meilleure gestion de l'extraction des changepoints

    Parameters:
    -----------
    y : array-like
        Série temporelle
    loss : str
        Type de loss ('biweight', 'huber', 'l2')
    beta : float
        Pénalité (si None, calculée automatiquement avec beta_scaling)
    beta_scaling : float
        Facteur de scaling pour beta (défaut: 4.0)
    K : float
        Paramètre K pour loss robuste (si None et recompute_K=False, calculé sur y complet)
    recompute_K : bool
        Si True, recalcule K à chaque étape sur les données partielles
    step : int
        Intervalle entre les calculs
    min_obs : int
        Minimum d'observations avant de commencer

    Les autres paramètres sont les fonctions importées du notebook.
    """
    n = len(y)
    y_list = list(y) if not isinstance(y, list) else y
    y_arr = np.array(y_list)

    # Calculer beta sur les données complètes
    if beta is None:
        beta_paper = compute_penalty_beta_func(y_arr, loss)
        beta = beta_scaling * beta_paper
        print(f"Beta paper: {beta_paper:.4f}, Beta utilisé (x{beta_scaling}): {beta:.4f}")

    # K global (utilisé si recompute_K=False)
    if K is None and not recompute_K and loss in ['huber', 'biweight']:
        K = compute_loss_bound_K_func(y_arr, loss)

    t_values = []
    most_recent_cp = []
    all_changepoints = []

    for t in tqdm(range(min_obs, n + 1, step), desc=f"Online analysis ({loss})"):
        y_partial = y_list[:t]
        y_partial_arr = np.array(y_partial)

        # Recalculer K si demandé
        if recompute_K and loss in ['huber', 'biweight']:
            K_current = compute_loss_bound_K_func(y_partial_arr, loss)
        else:
            K_current = K

        # Construire gamma_builder avec K_current
        if loss == 'huber':
            gamma_builder = lambda y_t, t_idx, K=K_current: gamma_builder_huber(y_t, K, t_idx)
        elif loss == 'biweight':
            gamma_builder = lambda y_t, t_idx, K=K_current: gamma_builder_biweight(y_t, K, t_idx)
        else:
            gamma_builder = lambda y_t, t_idx: gamma_builder_L2(y_t, t_idx)

        # Exécuter l'algorithme
        cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_partial, gamma_builder, beta)

        # Extraction des changepoints (corrigée)
        changepoints = extract_changepoints_robust(cp_tau)

        # Le plus récent
        recent_cp = changepoints[-1] if changepoints else 0

        t_values.append(t)
        most_recent_cp.append(recent_cp)
        all_changepoints.append(changepoints.copy())

    return {
        't_values': t_values,
        'most_recent_cp': most_recent_cp,
        'all_changepoints': all_changepoints,
        'params': {'beta': beta, 'K': K, 'loss': loss, 'beta_scaling': beta_scaling}
    }


def extract_changepoints_robust(cp_tau):
    """
    Extraction robuste des changepoints par backtracking.
    Filtre explicitement les valeurs <= 0.
    """
    n = len(cp_tau)
    changepoints = []
    idx = n - 1

    while idx > 0:
        tau = cp_tau[idx]
        if tau > 0:
            changepoints.append(tau)
        idx = tau

    # Tri et filtrage
    changepoints = [cp for cp in changepoints if cp > 0]
    changepoints.sort()
    return changepoints


def plot_most_recent_changepoint_comparison(
    online_result_original,
    online_result_fixed,
    true_changepoints=None,
    title=""
):
    """
    Compare les résultats de l'analyse originale et corrigée.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot original
    ax1 = axes[0]
    t_vals = online_result_original['t_values']
    recent_cp = online_result_original['most_recent_cp']
    ax1.step(t_vals, recent_cp, where='post', linewidth=2, color='black',
             label='Estimated most recent CP')
    if true_changepoints is not None:
        for cp in true_changepoints:
            ax1.axhline(y=cp, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Most Recent CP', fontsize=12)
    ax1.set_title(f'Original (beta scaling = {online_result_original["params"].get("beta_scaling", 12)})',
                  fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot fixed
    ax2 = axes[1]
    t_vals_f = online_result_fixed['t_values']
    recent_cp_f = online_result_fixed['most_recent_cp']
    ax2.step(t_vals_f, recent_cp_f, where='post', linewidth=2, color='blue',
             label='Estimated most recent CP (fixed)')
    if true_changepoints is not None:
        for cp in true_changepoints:
            ax2.axhline(y=cp, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Most Recent CP', fontsize=12)
    ax2.set_title(f'Fixed (beta scaling = {online_result_fixed["params"]["beta_scaling"]})',
                  fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{title}\nOnline Analysis Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


# =============================================================================
# Code à copier dans votre notebook pour tester
# =============================================================================

NOTEBOOK_CODE = '''
# =============================================================================
# ONLINE ANALYSIS CORRIGÉE - À copier dans votre notebook
# =============================================================================

def online_most_recent_changepoint_v2(
    y,
    loss: str = 'biweight',
    beta: float = None,
    beta_scaling: float = 4.0,  # CHANGÉ: de 12 à 4
    K: float = None,
    recompute_K: bool = True,   # NOUVEAU: recalcule K sur données partielles
    step: int = 50,
    min_obs: int = 100
):
    """
    Version corrigée de l'analyse online.

    Changements:
    1. beta_scaling=4 (au lieu de 12) - pénalité moins agressive
    2. recompute_K=True - K recalculé sur données partielles
    """
    n = len(y)
    y_list = list(y) if not isinstance(y, list) else y
    y_arr = np.array(y_list)

    # Calculer beta
    if beta is None:
        beta_paper = compute_penalty_beta(y_arr, loss)
        beta = beta_scaling * beta_paper
        print(f"Beta paper: {beta_paper:.4f}, Beta utilisé (x{beta_scaling}): {beta:.4f}")

    # K global si pas de recalcul
    K_global = None
    if K is None and not recompute_K and loss in ['huber', 'biweight']:
        K_global = compute_loss_bound_K(y_arr, loss)

    t_values = []
    most_recent_cp = []
    all_changepoints = []

    for t in tqdm(range(min_obs, n + 1, step), desc=f"Online analysis v2 ({loss})"):
        y_partial = y_list[:t]

        # Recalculer K si demandé
        if recompute_K and loss in ['huber', 'biweight']:
            K_current = compute_loss_bound_K(np.array(y_partial), loss)
        else:
            K_current = K_global if K is None else K

        # Gamma builder avec K_current
        if loss == 'huber':
            gamma_builder = lambda y_t, t_idx, K=K_current: gamma_builder_huber(y_t, K, t_idx)
        elif loss == 'biweight':
            gamma_builder = lambda y_t, t_idx, K=K_current: gamma_builder_biweight(y_t, K, t_idx)
        else:
            gamma_builder = lambda y_t, t_idx: gamma_builder_L2(y_t, t_idx)

        cp_tau, Qt_vals, _ = rfpop_algorithm1_main(y_partial, gamma_builder, beta)

        # Extraction robuste
        changepoints = []
        idx = t - 1
        while idx > 0:
            tau = cp_tau[idx]
            if tau > 0:
                changepoints.append(tau)
            idx = tau
        changepoints = sorted([cp for cp in changepoints if cp > 0])

        recent_cp = changepoints[-1] if changepoints else 0

        t_values.append(t)
        most_recent_cp.append(recent_cp)
        all_changepoints.append(changepoints.copy())

    return {
        't_values': t_values,
        'most_recent_cp': most_recent_cp,
        'all_changepoints': all_changepoints,
        'params': {'beta': beta, 'K': K_current, 'loss': loss, 'beta_scaling': beta_scaling}
    }


# =============================================================================
# TEST: Comparez les résultats avec différents paramètres
# =============================================================================

col = 'T10Y2Y'
y = df[col].dropna()
y = y[y.index > '2000']

print(f"Série: {col}, n = {len(y)}")

# Test avec différents beta_scaling
for scaling in [2, 4, 6, 8, 12]:
    print(f"\\n=== Beta scaling = {scaling} ===")

    result = online_most_recent_changepoint_v2(
        y.values,
        loss='biweight',
        beta_scaling=scaling,
        recompute_K=True,  # Important!
        step=20,
        min_obs=50
    )

    # Vérifier le nombre de chutes à 0
    drops_to_zero = sum(1 for i in range(1, len(result['most_recent_cp']))
                        if result['most_recent_cp'][i] == 0 and result['most_recent_cp'][i-1] > 0)

    print(f"Nombre de chutes à 0: {drops_to_zero}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(result['t_values'], result['most_recent_cp'], where='post', linewidth=2)
    ax.set_title(f'{col} - Beta scaling = {scaling}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Most Recent CP')
    ax.grid(True, alpha=0.3)
    plt.show()
'''

if __name__ == "__main__":
    print("Ce fichier contient les corrections pour l'analyse online.")
    print("\nCopiez le code suivant dans votre notebook:\n")
    print(NOTEBOOK_CODE)
