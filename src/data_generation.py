"""
IFRS 9 — Geração de Dados Sintéticos
======================================
Gera uma carteira de crédito fictícia com ~10.000 contratos.

Relações intencionalmente embutidas nos dados:
  - Score mais baixo      → maior probabilidade de default
  - DTI mais alto         → maior probabilidade de default
  - Mais inadimplências   → maior probabilidade de default
  - Garantia real forte   → menor LGD (maior recuperação)
  - LTV alto              → menor recuperação → maior LGD
  - Alta utilização       → menor CCF (pouco espaço para sacar mais)
"""

import os
import numpy as np
import pandas as pd


def generate_portfolio(n_loans: int = 10_000, seed: int = 42,
                       output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Gera carteira sintética completa com targets para PD, LGD e EAD.

    Parâmetros
    ----------
    n_loans    : número de contratos
    seed       : semente aleatória (reprodutibilidade)
    output_dir : pasta onde salvar 'portfolio.csv'

    Retorna
    -------
    DataFrame com todas as features e variáveis-alvo
    """
    rng = np.random.default_rng(seed)
    n = n_loans

    # =========================================================================
    # 1. CARACTERÍSTICAS DO CLIENTE
    # =========================================================================
    age = rng.integers(22, 72, n)

    # Renda anual: distribuição log-normal (assimetria positiva típica)
    income = np.exp(rng.normal(10.9, 0.65, n)).clip(15_000, 800_000)

    employment_years = rng.gamma(shape=3, scale=4, size=n).clip(0, 45)

    employment_type = rng.choice(
        ["CLT", "CNPJ", "Servidor_Publico", "Autonomo", "Aposentado"],
        p=[0.45, 0.18, 0.15, 0.14, 0.08],
        size=n,
    )

    # Score na ORIGINAÇÃO do contrato
    score_orig = rng.normal(650, 130, n).clip(200, 1000).astype(int)

    # Variação do score desde a originação (piora média de 15 pontos)
    score_current = (score_orig + rng.normal(-15, 75, n)).clip(200, 1000).astype(int)
    score_change = score_current - score_orig

    # =========================================================================
    # 2. CARACTERÍSTICAS DO PRODUTO / CONTRATO
    # =========================================================================
    loan_type = rng.choice(
        ["Imobiliario", "Veiculos", "Pessoal", "Cartao", "Empresarial"],
        p=[0.35, 0.20, 0.20, 0.15, 0.10],
        size=n,
    )

    # Valor do crédito varia por produto
    loan_amount = np.where(
        loan_type == "Imobiliario", np.exp(rng.normal(12.6, 0.5, n)),
        np.where(
            loan_type == "Veiculos",    np.exp(rng.normal(10.4, 0.4, n)),
            np.where(
                loan_type == "Empresarial", np.exp(rng.normal(11.5, 0.7, n)),
                                            np.exp(rng.normal(9.5,  0.6, n))
            )
        )
    ).clip(500, 3_000_000)

    # Prazo restante em meses
    remaining_months = np.where(
        loan_type == "Imobiliario", rng.integers(60, 360, n),
        np.where(
            loan_type == "Veiculos", rng.integers(12, 84, n),
                                     rng.integers(6,  48, n)
        )
    ).astype(int)

    # Percentual do saldo em aberto
    pct_used = rng.beta(3, 2, n)
    balance = loan_amount * pct_used

    # Produtos rotativos: cartão de crédito e limite empresarial
    is_revolving = np.isin(loan_type, ["Cartao", "Empresarial"])

    credit_limit = np.where(
        is_revolving,
        np.exp(rng.normal(9.5, 0.6, n)).clip(500, 500_000),
        loan_amount,
    )

    utilization = np.where(
        is_revolving, rng.beta(2, 3, n), pct_used
    ).clip(0.01, 0.99)

    drawn = np.where(is_revolving, credit_limit * utilization, balance)
    undrawn = np.where(is_revolving, credit_limit - drawn, 0.0)

    # =========================================================================
    # 3. INDICADORES DE RISCO
    # =========================================================================
    # DTI = parcela estimada / renda mensal
    monthly_payment = balance * 0.035
    dti = (monthly_payment / (income / 12)).clip(0, 2.0)

    num_delinq = rng.choice([0, 1, 2, 3, 4], p=[0.65, 0.19, 0.09, 0.05, 0.02], size=n)
    dpd = np.where(
        num_delinq == 0, 0,
        rng.choice([15, 30, 45, 60, 90, 120], p=[0.30, 0.30, 0.15, 0.10, 0.10, 0.05], size=n),
    )

    # Colateral e LTV
    collateral = np.where(
        loan_type == "Imobiliario", loan_amount / rng.uniform(0.65, 0.90, n),
        np.where(
            loan_type == "Veiculos", loan_amount / rng.uniform(0.75, 0.95, n),
                                     loan_amount * rng.uniform(0.0,  0.20, n)
        )
    )
    ltv = np.where(collateral > 100, balance / collateral, 1.0).clip(0, 2.0)

    interest_rate = np.where(
        loan_type == "Cartao",      rng.uniform(0.15, 0.45, n),
        np.where(
            loan_type == "Imobiliario", rng.uniform(0.06, 0.13, n),
            np.where(
                loan_type == "Veiculos", rng.uniform(0.08, 0.20, n),
                                         rng.uniform(0.12, 0.35, n)
            )
        )
    )

    # =========================================================================
    # 4. TARGET — DEFAULT FLAG  (variável resposta do modelo de PD)
    # =========================================================================
    # Equação logística que embute relações realistas:
    #   score  ↑  →  logit ↓  →  PD ↓
    #   DTI    ↑  →  logit ↑  →  PD ↑
    #   atrasos↑  →  logit ↑  →  PD ↑
    logit_pd = (
        -4.5
        - 0.006 * (score_current - 500)        # score melhor → menos risco
        + 0.8   * (dti > 0.40).astype(float)   # DTI moderadamente alto
        + 1.5   * (dti > 0.60).astype(float)   # DTI muito alto
        + 0.40  * num_delinq                    # cada inadimplência passada
        + 0.02  * dpd                           # cada dia em atraso
        - 0.02  * employment_years              # estabilidade  emprego
        - 0.40  * (employment_type == "Servidor_Publico").astype(float)
        + 0.35  * (employment_type == "Autonomo").astype(float)
        + 0.40  * (ltv > 0.85).astype(float)   # LTV elevado
        + 0.60  * (ltv > 1.00).astype(float)   # LTV acima de 100%
        + rng.normal(0, 0.30, n)               # ruído idiossincrático
    )

    pd_true = (1 / (1 + np.exp(-logit_pd))).clip(1e-4, 1 - 1e-4)
    default_flag = (rng.random(n) < pd_true).astype(int)

    # =========================================================================
    # 5. TARGET — RECOVERY RATE → LGD  (variável resposta do modelo de LGD)
    # =========================================================================
    # Taxa de recuperação depende principalmente da qualidade da garantia
    rec_base = np.where(
        loan_type == "Imobiliario", 0.72,
        np.where(loan_type == "Veiculos",    0.52,
        np.where(loan_type == "Empresarial", 0.40,
        np.where(loan_type == "Pessoal",     0.22,
                                             0.12)))  # Cartão: recuperação baixíssima
    )

    rec_logit = (
        np.log(rec_base / (1 - rec_base))   # âncora no valor esperado por tipo
        - 0.5  * np.maximum(ltv - 0.70, 0)  # garantia "subaquática" → menos recuperação
        - 0.3  * (loan_type == "Pessoal").astype(float)
        + rng.normal(0, 0.4, n)
    )
    recovery_rate = (1 / (1 + np.exp(-rec_logit))).clip(1e-4, 1 - 1e-4)
    lgd_true = 1 - recovery_rate             # LGD = 1 − Recovery Rate

    # =========================================================================
    # 6. TARGET — EAD / CCF  (variável resposta do modelo de EAD)
    # =========================================================================
    # Credit Conversion Factor: % do limite não sacado que será usado antes do default
    # Clientes com alta utilização têm pouco espaço para sacar → CCF menor
    ccf_logit = (
        0.3
        + 0.5  * utilization.clip(0, 1)
        + rng.normal(0, 0.3, n)
    )
    ccf_true = (1 / (1 + np.exp(-ccf_logit))).clip(1e-4, 1 - 1e-4)

    ead_true = np.where(
        is_revolving,
        drawn + ccf_true * undrawn,   # EAD = sacado + CCF × não-sacado
        balance                        # não-rotativos: EAD = saldo devedor
    )

    # =========================================================================
    # 7. STAGING IFRS 9
    # =========================================================================
    # ASRC = Aumento Significativo do Risco de Crédito
    sicr = (
        (score_change < -100) |                           # queda expressiva no score
        (dpd >= 30) |                                     # 30+ dias em atraso
        ((dti > 0.55) & (num_delinq > 0))                 # risco combinado alto
    )

    stage = np.where(default_flag == 1, 3,
                     np.where(sicr, 2, 1)).astype(int)

    # =========================================================================
    # 8. MONTAR DATAFRAME
    # =========================================================================
    df = pd.DataFrame({
        "loan_id": [f"LN{i:06d}" for i in range(1, n + 1)],
        # --- Cliente ---
        "age":               age,
        "annual_income":     income.round(2),
        "employment_years":  employment_years.round(1),
        "employment_type":   employment_type,
        "credit_score_orig":    score_orig,
        "credit_score_current": score_current,
        "score_change":         score_change,
        # --- Produto ---
        "loan_type":            loan_type,
        "loan_amount":          loan_amount.round(2),
        "outstanding_balance":  balance.round(2),
        "credit_limit":         credit_limit.round(2),
        "drawn_balance":        drawn.round(2),
        "undrawn_balance":      undrawn.round(2),
        "utilization_rate":     utilization.round(4),
        "remaining_months":     remaining_months,
        "interest_rate":        interest_rate.round(4),
        # --- Risco ---
        "debt_to_income":       dti.round(4),
        "ltv_ratio":            ltv.round(4),
        "collateral_value":     collateral.round(2),
        "num_delinquencies":    num_delinq,
        "days_past_due":        dpd,
        # --- Targets ---
        "default_flag":    default_flag,            # PD target
        "recovery_rate":   recovery_rate.round(4),  # intermediário
        "lgd_true":        lgd_true.round(4),       # LGD target
        "ead_true":        ead_true.round(2),        # EAD target
        "ccf_true":        ccf_true.round(4),        # CCF target (rotativos)
        "pd_true":         pd_true.round(6),         # PD "real" (latente)
        # --- IFRS 9 ---
        "stage": stage,
    })

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "portfolio.csv"), index=False)

    return df


def print_portfolio_summary(df: pd.DataFrame) -> None:
    """Imprime estatísticas descritivas da carteira gerada."""
    total = len(df)
    n_default = df["default_flag"].sum()
    print(f"\n{'='*55}")
    print("  RESUMO DA CARTEIRA GERADA")
    print(f"{'='*55}")
    print(f"  Contratos totais       : {total:>10,}")
    print(f"  Exposição total (EAD)  : R$ {df['ead_true'].sum():>14,.2f}")
    print(f"  Taxa de default        : {n_default/total:>10.2%}")
    print(f"  LGD médio (defaults)   : "
          f"{df.loc[df['default_flag']==1, 'lgd_true'].mean():>10.2%}")

    print(f"\n  {'Tipo':20s}  {'N':>6}  {'% Default':>10}")
    print(f"  {'-'*40}")
    for lt, grp in df.groupby("loan_type"):
        pct = grp["default_flag"].mean()
        print(f"  {lt:20s}  {len(grp):>6,}  {pct:>10.2%}")

    print(f"\n  Stage  {'N':>6}  {'% Carteira':>12}  {'ECL esperada':>16}")
    print(f"  {'-'*44}")
    for s, grp in df.groupby("stage"):
        pct = len(grp) / total
        ecl_est = (grp["pd_true"] * grp["lgd_true"] * grp["ead_true"]).sum()
        print(f"  {s:>5}  {len(grp):>6,}  {pct:>12.2%}  R$ {ecl_est:>13,.2f}")

    print(f"{'='*55}\n")
