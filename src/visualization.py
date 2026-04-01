"""
IFRS 9 — Visualizações
========================
Gera todos os gráficos de análise, validação e resultados dos modelos.

Gráficos gerados:
  1.  Distribuição do Score de Crédito (por inadimplência)
  2.  Composição da Carteira por Stage
  3.  Curva ROC — Modelo de PD
  4.  Distribuição dos Scores de PD (bons vs. maus)
  5.  Curva KS — Modelo de PD
  6.  Calibração do Modelo de PD (previsto vs. real)
  7.  Distribuição do LGD (previsto vs. real)
  8.  LGD por Tipo de Produto
  9.  Scatter EAD previsto vs. real
  10. Distribuição do CCF (rotativos)
  11. ECL por Stage (barras)
  12. Distribuição da ECL individual
  13. Análise de Sensibilidade (heatmap)
  14. Matriz de Correlação das Features de PD
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # Backend sem display (salva em arquivo)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# Estilo global
PALETTE   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
STYLE     = "seaborn-v0_8-whitegrid"
DPI       = 120
FONT_SIZE = 11

plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": DPI})


def _save(fig: plt.Figure, folder: str, filename: str) -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Salvo: {path}")


# ---------------------------------------------------------------------------
# 1. Score de crédito por inadimplência
# ---------------------------------------------------------------------------
def plot_score_distribution(df: pd.DataFrame,
                             plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição do Score de Crédito", fontweight="bold")

    for ax, col, title in zip(axes,
                               ["credit_score_orig", "credit_score_current"],
                               ["Score na Originação", "Score Atual"]):
        for flag, label, color in [(0, "Adimplente", "#2ca02c"), (1, "Default", "#d62728")]:
            data = df.loc[df["default_flag"] == flag, col]
            ax.hist(data, bins=40, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Densidade")
        ax.legend()

    fig.tight_layout()
    _save(fig, plots_dir, "01_score_distribution.png")


# ---------------------------------------------------------------------------
# 2. Composição da carteira por Stage
# ---------------------------------------------------------------------------
def plot_staging(df: pd.DataFrame,
                 plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Composição da Carteira — IFRS 9 Staging", fontweight="bold")

    stage_counts = df["stage"].value_counts().sort_index()
    labels = [f"Stage {s}" for s in stage_counts.index]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    # Pizza: número de contratos
    axes[0].pie(stage_counts, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("% de Contratos por Stage")

    # Barra: EAD por stage
    ead_by_stage = df.groupby("stage")["ead_true"].sum() / 1e6
    bars = axes[1].bar([f"Stage {s}" for s in ead_by_stage.index],
                        ead_by_stage.values, color=colors, edgecolor="white")
    axes[1].set_title("Exposição Total (EAD) por Stage — R$ Milhões")
    axes[1].set_ylabel("R$ Milhões")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f"R${bar.get_height():.1f}M", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, plots_dir, "02_staging.png")


# ---------------------------------------------------------------------------
# 3 & 4. ROC Curve e Score Distribution (PD)
# ---------------------------------------------------------------------------
def plot_pd_performance(pd_results: dict,
                         plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Modelo de PD — Desempenho", fontweight="bold")

    # Curva ROC
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC=0.50)")

    colors_model = {"logistic": "#1f77b4", "gbm": "#ff7f0e"}
    for name, res in pd_results.items():
        fpr, tpr = res["fpr"], res["tpr"]
        auc  = res["auc_test"]
        gini = res["gini"]
        label = f"{name.upper()} (AUC={auc:.3f}, Gini={gini:.3f})"
        ax_roc.plot(fpr, tpr, lw=2, color=colors_model[name], label=label)

    ax_roc.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend(fontsize=9)

    # Distribuição dos scores: bons vs maus
    ax_dist = axes[1]
    best = pd_results["gbm"]
    scores = best["prob_test"]
    y_test = best["y_test"]
    ks     = best["ks"]

    for flag, label, color in [(0, "Adimplente", "#2ca02c"), (1, "Default", "#d62728")]:
        mask = y_test == flag
        ax_dist.hist(scores[mask], bins=30, alpha=0.6, color=color,
                     label=label, density=True)

    ax_dist.set_xlabel("Score de PD (GBM)")
    ax_dist.set_ylabel("Densidade")
    ax_dist.set_title(f"Distribuição dos Scores de PD  (KS={ks:.3f})")
    ax_dist.legend()

    fig.tight_layout()
    _save(fig, plots_dir, "03_pd_performance.png")


# ---------------------------------------------------------------------------
# 5. Curva KS
# ---------------------------------------------------------------------------
def plot_ks_curve(pd_results: dict,
                  plots_dir: str = "outputs/plots") -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Curva KS — Modelo de PD (GBM)", fontweight="bold")

    best = pd_results["gbm"]
    y_test = best["y_test"]
    scores = best["prob_test"]

    fpr, tpr, thresholds = roc_curve(y_test, scores)
    ks_vals = tpr - fpr
    ks_idx  = np.argmax(ks_vals)

    ax.plot(thresholds[::-1], fpr[::-1],  color="#1f77b4", lw=2, label="Cumulativo Bons")
    ax.plot(thresholds[::-1], tpr[::-1],  color="#d62728", lw=2, label="Cumulativo Maus")
    ax.axvline(thresholds[ks_idx], color="gray", ls="--", lw=1.5,
               label=f"KS={best['ks']:.3f} @ threshold={thresholds[ks_idx]:.3f}")

    ax.set_xlabel("Threshold de Score")
    ax.set_ylabel("% Cumulativo")
    ax.set_title("")
    ax.legend()
    fig.tight_layout()
    _save(fig, plots_dir, "04_ks_curve.png")


# ---------------------------------------------------------------------------
# 6. Calibração do Modelo de PD
# ---------------------------------------------------------------------------
def plot_pd_calibration(pd_results: dict,
                         plots_dir: str = "outputs/plots") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Calibração do Modelo de PD", fontweight="bold")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibração perfeita")

    colors_model = {"logistic": "#1f77b4", "gbm": "#ff7f0e"}
    for name, res in pd_results.items():
        prob_true, prob_pred = calibration_curve(
            res["y_test"], res["prob_test"], n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, "o-", lw=2,
                color=colors_model[name], label=name.upper(), markersize=6)

    ax.set_xlabel("PD Previsto (média do bin)")
    ax.set_ylabel("Taxa de Default Real (média do bin)")
    ax.set_title("")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, plots_dir, "05_pd_calibration.png")


# ---------------------------------------------------------------------------
# 7. Distribuição do LGD
# ---------------------------------------------------------------------------
def plot_lgd_distribution(df: pd.DataFrame,
                           lgd_results: dict,
                           plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Modelo de LGD — Loss Given Default", fontweight="bold")

    # Distribuição real por tipo de produto
    ax1 = axes[0]
    loan_types = df["loan_type"].unique()
    data_lgd = [df.loc[df["loan_type"] == lt, "lgd_true"].values
                for lt in loan_types]
    bp = ax1.boxplot(data_lgd, labels=loan_types, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel("Tipo de Produto")
    ax1.set_ylabel("LGD Real")
    ax1.set_title("LGD por Tipo de Produto (verdadeiro)")
    ax1.tick_params(axis="x", rotation=15)

    # Previsto vs Real (GBM)
    ax2 = axes[1]
    best = lgd_results["gbm"]
    ax2.scatter(best["y_test"], best["pred_test"],
                alpha=0.3, s=10, color="#1f77b4")
    lim = max(best["y_test"].max(), best["pred_test"].max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfeito")
    ax2.set_xlabel("LGD Real")
    ax2.set_ylabel("LGD Previsto (GBM)")
    ax2.set_title(f"GBM: R²={best['r2']:.3f}  MAE={best['mae']:.3f}")
    ax2.legend()
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, plots_dir, "06_lgd_model.png")


# ---------------------------------------------------------------------------
# 8. Modelo de EAD/CCF
# ---------------------------------------------------------------------------
def plot_ead_model(df: pd.DataFrame,
                   ead_results: dict,
                   plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Modelo de EAD / CCF — Produtos Rotativos", fontweight="bold")

    # Distribuição do CCF real vs previsto
    ax1 = axes[0]
    best = ead_results["gbm"]
    ax1.hist(best["y_test"],    bins=30, alpha=0.6, color="#1f77b4",
             label="CCF Real", density=True)
    ax1.hist(best["pred_test"], bins=30, alpha=0.6, color="#ff7f0e",
             label="CCF Previsto (GBM)", density=True)
    ax1.set_xlabel("CCF")
    ax1.set_ylabel("Densidade")
    ax1.set_title(f"Distribuição do CCF  (R²={best['r2']:.3f})")
    ax1.legend()

    # Scatter CCF previsto vs real
    ax2 = axes[1]
    ax2.scatter(best["y_test"], best["pred_test"],
                alpha=0.3, s=10, color="#2ca02c")
    lim = max(best["y_test"].max(), best["pred_test"].max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfeito")
    ax2.set_xlabel("CCF Real")
    ax2.set_ylabel("CCF Previsto")
    ax2.set_title(f"GBM: R²={best['r2']:.3f}  MAE={best['mae']:.3f}")
    ax2.legend()

    fig.tight_layout()
    _save(fig, plots_dir, "07_ead_model.png")


# ---------------------------------------------------------------------------
# 9. ECL por Stage
# ---------------------------------------------------------------------------
def plot_ecl_results(portfolio: pd.DataFrame,
                     plots_dir: str = "outputs/plots") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Resultado da ECL — IFRS 9", fontweight="bold")

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    stage_labels = ["Stage 1\n(ECL 12m)", "Stage 2\n(ECL Lifetime)", "Stage 3\n(Lifetime)"]

    # ECL total por Stage
    ax1 = axes[0]
    ecl_by_stage = portfolio.groupby("stage")["ecl_final"].sum() / 1e6
    bars = ax1.bar(stage_labels[:len(ecl_by_stage)],
                   ecl_by_stage.values, color=colors[:len(ecl_by_stage)],
                   edgecolor="white")
    ax1.set_ylabel("ECL (R$ Milhões)")
    ax1.set_title("ECL Total por Stage")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"R${bar.get_height():.1f}M",
                 ha="center", va="bottom", fontsize=9)

    # Índice de cobertura por Stage
    ax2 = axes[1]
    cov_by_stage = portfolio.groupby("stage").apply(
        lambda g: g["ecl_final"].sum() / g["ead_pred"].sum()
    )
    bars2 = ax2.bar(stage_labels[:len(cov_by_stage)],
                    cov_by_stage.values * 100,
                    color=colors[:len(cov_by_stage)], edgecolor="white")
    ax2.set_ylabel("Índice de Cobertura (%)")
    ax2.set_title("ECL / EAD por Stage")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"{bar.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=9)

    # ECL por tipo de produto
    ax3 = axes[2]
    ecl_by_type = portfolio.groupby("loan_type")["ecl_final"].sum() / 1e6
    ecl_by_type = ecl_by_type.sort_values(ascending=True)
    ax3.barh(ecl_by_type.index, ecl_by_type.values, color=PALETTE[:len(ecl_by_type)],
             edgecolor="white")
    ax3.set_xlabel("ECL (R$ Milhões)")
    ax3.set_title("ECL por Tipo de Produto")

    fig.tight_layout()
    _save(fig, plots_dir, "08_ecl_results.png")


# ---------------------------------------------------------------------------
# 10. Análise de Sensibilidade
# ---------------------------------------------------------------------------
def plot_sensitivity(sensitivity_df: pd.DataFrame,
                     plots_dir: str = "outputs/plots") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Análise de Sensibilidade da ECL\n(choque em PD e LGD)",
                 fontweight="bold")

    pivot = sensitivity_df.pivot(
        index="lgd_multiplier",
        columns="pd_multiplier",
        values="delta_pct",
    )
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.5, cbar_kws={"label": "Δ ECL (%)"})
    ax.set_xlabel("Multiplicador de PD")
    ax.set_ylabel("Multiplicador de LGD")
    ax.set_title("")

    fig.tight_layout()
    _save(fig, plots_dir, "09_sensitivity.png")


# ---------------------------------------------------------------------------
# 11. Painel de Feature Importance (PD GBM)
# ---------------------------------------------------------------------------
def plot_feature_importance(pd_results: dict,
                             plots_dir: str = "outputs/plots") -> None:
    try:
        pipe     = pd_results["gbm"]["pipeline"]
        model    = pipe.named_steps["model"]
        prep     = pipe.named_steps["preprocessor"]

        # Nomes das features pós-encoding
        num_names = prep.transformers_[0][2]
        cat_enc   = prep.transformers_[1][1]
        cat_names = cat_enc.get_feature_names_out(prep.transformers_[1][2]).tolist()
        all_names = num_names + cat_names

        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]  # top 15

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([all_names[i] for i in idx], importances[idx],
                color="#1f77b4", edgecolor="white")
        ax.set_xlabel("Feature Importance (GBM)")
        ax.set_title("Top Features — Modelo de PD (GBM)")
        fig.tight_layout()
        _save(fig, plots_dir, "10_feature_importance.png")
    except Exception:
        pass   # não crítico se falhar


# ---------------------------------------------------------------------------
# Função orquestradora
# ---------------------------------------------------------------------------
def generate_all_plots(df: pd.DataFrame,
                        pd_results: dict,
                        lgd_results: dict,
                        ead_results: dict,
                        portfolio: pd.DataFrame,
                        sensitivity_df: pd.DataFrame,
                        plots_dir: str = "outputs/plots") -> None:
    """Gera todos os gráficos do projeto."""
    print("\n" + "=" * 55)
    print("  GERANDO GRÁFICOS")
    print("=" * 55)

    plot_score_distribution(df, plots_dir)
    plot_staging(df, plots_dir)
    plot_pd_performance(pd_results, plots_dir)
    plot_ks_curve(pd_results, plots_dir)
    plot_pd_calibration(pd_results, plots_dir)
    plot_lgd_distribution(df[df["default_flag"] == 1].copy(), lgd_results, plots_dir)
    plot_ead_model(df, ead_results, plots_dir)
    plot_ecl_results(portfolio, plots_dir)
    plot_sensitivity(sensitivity_df, plots_dir)
    plot_feature_importance(pd_results, plots_dir)

    print(f"\n  Todos os gráficos salvos em: {plots_dir}/")
