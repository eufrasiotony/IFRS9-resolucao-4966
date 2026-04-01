"""
IFRS 9 — Cálculo da ECL (Expected Credit Loss)
================================================
Combina os três componentes — PD, LGD e EAD — para calcular a provisão.

Fórmula base:
    ECL = PD × LGD × EAD × Fator de Desconto

Aplicação por estágio:
    Stage 1 → ECL 12 meses    : PD_12m × LGD × EAD × DF
    Stage 2 → ECL Lifetime    : Σ_t [ PD_marginal_t × LGD × EAD_t × DF_t ]
    Stage 3 → ECL Lifetime    : LGD × EAD (PD = 100%, já em default)

Fator de Desconto:
    DF_t = 1 / (1 + EIR)^t   onde EIR = taxa de juros efetiva do contrato
    Para simplificação didática, usamos uma taxa de desconto fixa por contrato.

Resultado esperado:
    - ECL por contrato
    - ECL agregado por estágio (Stage 1 / 2 / 3)
    - Índice de cobertura = ECL / EAD
    - Concentração de risco por tipo de produto
"""

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cálculo do desconto
# ---------------------------------------------------------------------------
def discount_factor(rate: float, period_years: float) -> float:
    """
    Fator de valor presente para um único período.
        DF = 1 / (1 + r)^t
    """
    return 1 / ((1 + rate) ** period_years)


def lifetime_ecl_vectorized(pd_12m: np.ndarray,
                              lgd: np.ndarray,
                              ead: np.ndarray,
                              interest_rate: np.ndarray,
                              remaining_months: np.ndarray) -> np.ndarray:
    """
    Calcula a ECL lifetime de forma vetorizada.

    Abordagem simplificada (didática):
      Para cada contrato, itera ano a ano até o vencimento.
      PD marginal no ano t = P(default no ano t | sobreviveu até t-1)
                           = (1 − PD_12m)^(t-1) × PD_12m

      ECL_lifetime = Σ_{t=1}^{T}  PD_marginal_t × LGD × EAD × DF_t

    Assume EAD e LGD constantes ao longo do tempo (simplificação).
    """
    T_years = np.maximum(remaining_months, 1) / 12.0
    T_max   = int(np.ceil(T_years.max()))

    ecl = np.zeros(len(pd_12m))

    for t in range(1, T_max + 1):
        # Apenas para contratos com prazo >= t
        mask = T_years >= t

        # PD marginal: probabilidade de defaultar exatamente no ano t
        pd_marginal = (1 - pd_12m) ** (t - 1) * pd_12m

        # Fator de desconto com taxa EIR do contrato
        df_t = 1 / ((1 + interest_rate) ** t)

        ecl += np.where(mask, pd_marginal * lgd * ead * df_t, 0)

    return ecl


# ---------------------------------------------------------------------------
# Cálculo principal
# ---------------------------------------------------------------------------
def calculate_ecl(df: pd.DataFrame,
                  pd_scores: pd.DataFrame,
                  lgd_scores: pd.DataFrame,
                  ead_scores: pd.DataFrame,
                  output_dir: str = "outputs") -> pd.DataFrame:
    """
    Calcula a ECL para cada contrato da carteira.

    Parâmetros
    ----------
    df          : carteira original
    pd_scores   : ['loan_id', 'pd_12m', 'pd_lifetime']
    lgd_scores  : ['loan_id', 'lgd_pred']
    ead_scores  : ['loan_id', 'ead_pred']
    output_dir  : diretório de saída

    Retorna
    -------
    DataFrame com todos os componentes e a ECL calculada
    """
    print("\n" + "=" * 55)
    print("  CÁLCULO DA ECL — Expected Credit Loss")
    print("=" * 55)

    # Merge de todos os scores
    portfolio = (
        df[["loan_id", "loan_type", "stage", "remaining_months",
            "interest_rate", "outstanding_balance", "default_flag"]]
        .merge(pd_scores, on="loan_id")
        .merge(lgd_scores, on="loan_id")
        .merge(ead_scores, on="loan_id")
    )

    n = len(portfolio)
    pd_12m     = portfolio["pd_12m"].values
    pd_lifetime = portfolio["pd_lifetime"].values
    lgd        = portfolio["lgd_pred"].values
    ead        = portfolio["ead_pred"].values
    eir        = portfolio["interest_rate"].values
    rem_months = portfolio["remaining_months"].values
    stage      = portfolio["stage"].values

    # -----------------------------------------------------------------------
    # ECL 12 meses (Stage 1)
    # -----------------------------------------------------------------------
    df_12m = 1 / ((1 + eir) ** (1.0))          # desconto de 1 ano
    ecl_12m = pd_12m * lgd * ead * df_12m

    # -----------------------------------------------------------------------
    # ECL Lifetime (Stage 2 e 3)
    # -----------------------------------------------------------------------
    print("\n  Calculando ECL Lifetime (pode demorar alguns segundos)...")
    ecl_lifetime = lifetime_ecl_vectorized(pd_12m, lgd, ead, eir, rem_months)

    # -----------------------------------------------------------------------
    # Stage 3: crédito deteriorado → PD = 100% (já em default)
    # -----------------------------------------------------------------------
    ecl_stage3 = lgd * ead   # sem desconto (perda imediata)

    # -----------------------------------------------------------------------
    # ECL final: combina por estágio
    # -----------------------------------------------------------------------
    ecl_final = np.where(
        stage == 1, ecl_12m,
        np.where(stage == 2, ecl_lifetime, ecl_stage3)
    )

    portfolio["ecl_12m"]     = ecl_12m.round(2)
    portfolio["ecl_lifetime"] = ecl_lifetime.round(2)
    portfolio["ecl_final"]   = ecl_final.round(2)
    portfolio["coverage_ratio"] = (ecl_final / np.maximum(ead, 1)).round(4)

    # -----------------------------------------------------------------------
    # Relatório consolidado por estágio
    # -----------------------------------------------------------------------
    print("\n  Resultado da ECL por Estágio:")
    print(f"\n  {'Stage':<8} {'Contratos':>10} {'EAD Total':>18} "
          f"{'ECL Total':>18} {'Cobertura':>10}")
    print(f"  {'-'*68}")

    report_rows = []
    for s in [1, 2, 3]:
        mask = stage == s
        n_s   = mask.sum()
        ead_s = ead[mask].sum()
        ecl_s = ecl_final[mask].sum()
        cov_s = ecl_s / ead_s if ead_s > 0 else 0
        print(f"  {s:<8} {n_s:>10,} {ead_s:>18,.2f} {ecl_s:>18,.2f} {cov_s:>10.2%}")
        report_rows.append({
            "stage": s,
            "n_loans": n_s,
            "ead_total": round(ead_s, 2),
            "ecl_total": round(ecl_s, 2),
            "coverage_ratio": round(cov_s, 4),
        })

    # Total
    ead_total = ead.sum()
    ecl_total = ecl_final.sum()
    cov_total = ecl_total / ead_total
    print(f"  {'TOTAL':<8} {n:>10,} {ead_total:>18,.2f} "
          f"{ecl_total:>18,.2f} {cov_total:>10.2%}")

    print(f"\n  Índice de Cobertura Total: {cov_total:.2%}")

    # -----------------------------------------------------------------------
    # Breakdown por tipo de produto
    # -----------------------------------------------------------------------
    print("\n  ECL por Tipo de Produto:")
    print(f"\n  {'Produto':<15} {'Contratos':>10} {'ECL Total':>18} {'Cobertura':>10}")
    print(f"  {'-'*55}")
    for lt, grp in portfolio.groupby("loan_type"):
        ecl_lt = grp["ecl_final"].sum()
        ead_lt = grp["ead_pred"].sum()
        cov_lt = ecl_lt / ead_lt if ead_lt > 0 else 0
        print(f"  {lt:<15} {len(grp):>10,} {ecl_lt:>18,.2f} {cov_lt:>10.2%}")

    # -----------------------------------------------------------------------
    # Salvar resultados
    # -----------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Relatório de estágios
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(os.path.join(output_dir, "ecl_report_stages.csv"), index=False)

    # Carteira completa com ECL
    portfolio.to_csv("data/processed/portfolio_final.csv", index=False)

    print(f"\n  Arquivos salvos:")
    print(f"    data/processed/portfolio_final.csv")
    print(f"    {output_dir}/ecl_report_stages.csv")

    return portfolio


# ---------------------------------------------------------------------------
# Utilitários analíticos
# ---------------------------------------------------------------------------
def ecl_sensitivity_analysis(portfolio: pd.DataFrame,
                               pd_shock: float = 0.2,
                               lgd_shock: float = 0.1) -> pd.DataFrame:
    """
    Análise de sensibilidade da ECL a choques nos parâmetros.

    Modela cenários adversos aumentando PD e LGD em percentuais fixos.
    Útil para estresse do balanço patrimonial (stress testing).

    Parâmetros
    ----------
    pd_shock  : choque percentual na PD  (ex: 0.20 = +20%)
    lgd_shock : choque percentual na LGD (ex: 0.10 = +10%)
    """
    base_ecl   = portfolio["ecl_final"].sum()
    ead_total  = portfolio["ead_pred"].sum()

    scenarios = []
    for pd_mult in [1.0, 1 + pd_shock * 0.5, 1 + pd_shock, 1 + pd_shock * 2]:
        for lgd_mult in [1.0, 1 + lgd_shock * 0.5, 1 + lgd_shock]:
            ecl_stress = (
                (portfolio["ecl_final"] * pd_mult * lgd_mult)
                .clip(upper=portfolio["ead_pred"])
                .sum()
            )
            scenarios.append({
                "pd_multiplier":   round(pd_mult, 2),
                "lgd_multiplier":  round(lgd_mult, 2),
                "ecl_stress":      round(ecl_stress, 2),
                "delta_ecl":       round(ecl_stress - base_ecl, 2),
                "delta_pct":       round((ecl_stress / base_ecl - 1) * 100, 2),
                "coverage_ratio":  round(ecl_stress / ead_total, 4),
            })

    return pd.DataFrame(scenarios)
