"""
IFRS 9 — Orquestrador Principal
=================================
Executa o pipeline completo de modelagem de risco de crédito:

    1. Gera carteira sintética (10.000 contratos)
    2. Treina modelo de PD  → AUC, Gini, KS
    3. Treina modelo de LGD → R², MAE, Spearman
    4. Treina modelo de EAD → R², MAE (CCF)
    5. Calcula ECL por estágio (Stage 1 / 2 / 3)
    6. Analisa sensibilidade (stress test)
    7. Gera todos os gráficos em outputs/plots/

Execute com:
    python main.py

Saídas:
    data/raw/portfolio.csv              — carteira gerada
    data/processed/portfolio_final.csv  — carteira com scores e ECL
    outputs/models/                     — modelos treinados (.pkl)
    outputs/plots/                      — gráficos de validação
    outputs/ecl_report_stages.csv       — resumo de ECL por stage
"""

import sys
import time
import os
import json

# Garante que o diretório raiz está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generation import generate_portfolio, print_portfolio_summary
from src.pd_model        import train_pd_model
from src.lgd_model       import train_lgd_model
from src.ead_model       import train_ead_model
from src.ecl_calculator  import calculate_ecl, ecl_sensitivity_analysis
from src.visualization   import generate_all_plots

OUTPUT_DIR = "outputs"
N_LOANS    = 10_000


def banner(text: str) -> None:
    w = 55
    print("\n" + "█" * w)
    print(f"█  {text:<{w-4}}█")
    print("█" * w)


def main() -> None:
    t0 = time.time()

    banner("IFRS 9 — Modelagem de Risco de Crédito")
    print("""
  Este projeto demonstra a construção dos três modelos
  exigidos pelo IFRS 9 / CPC 48 para cálculo da ECL:

    PD  — Probability of Default
    LGD — Loss Given Default
    EAD — Exposure at Default

  ECL = PD × LGD × EAD × Fator de Desconto
    """)

    # =========================================================================
    # STEP 1 — Geração de Dados Sintéticos
    # =========================================================================
    banner("STEP 1 — Geração de Dados")
    print(f"\n  Gerando carteira com {N_LOANS:,} contratos sintéticos...")
    df = generate_portfolio(n_loans=N_LOANS, output_dir="data/raw")
    print_portfolio_summary(df)

    # =========================================================================
    # STEP 2 — Modelo de PD
    # =========================================================================
    banner("STEP 2 — Modelo de PD")
    pd_results, pd_scores = train_pd_model(df, output_dir=OUTPUT_DIR)

    print(f"\n  Resumo de desempenho:")
    print(f"  {'Modelo':<12} {'AUC':>8} {'Gini':>8} {'KS':>8}")
    print(f"  {'-'*40}")
    for name, res in pd_results.items():
        print(f"  {name.upper():<12} {res['auc_test']:>8.4f} "
              f"{res['gini']:>8.4f} {res['ks']:>8.4f}")

    # =========================================================================
    # STEP 3 — Modelo de LGD
    # =========================================================================
    banner("STEP 3 — Modelo de LGD")
    lgd_results, lgd_scores = train_lgd_model(df, output_dir=OUTPUT_DIR)

    print(f"\n  Resumo de desempenho:")
    print(f"  {'Modelo':<12} {'R²':>8} {'MAE':>8} {'Spearman':>10}")
    print(f"  {'-'*42}")
    for name, res in lgd_results.items():
        print(f"  {name.upper():<12} {res['r2']:>8.4f} "
              f"{res['mae']:>8.4f} {res['spearman']:>10.4f}")

    # =========================================================================
    # STEP 4 — Modelo de EAD
    # =========================================================================
    banner("STEP 4 — Modelo de EAD / CCF")
    ead_results, ead_scores = train_ead_model(df, output_dir=OUTPUT_DIR)

    print(f"\n  Resumo de desempenho (CCF — produtos rotativos):")
    print(f"  {'Modelo':<12} {'R²':>8} {'MAE':>8} {'Spearman':>10}")
    print(f"  {'-'*42}")
    for name, res in ead_results.items():
        print(f"  {name.upper():<12} {res['r2']:>8.4f} "
              f"{res['mae']:>8.4f} {res['spearman']:>10.4f}")

    # =========================================================================
    # STEP 5 — Cálculo da ECL
    # =========================================================================
    banner("STEP 5 — Cálculo da ECL")
    portfolio = calculate_ecl(df, pd_scores, lgd_scores, ead_scores,
                               output_dir=OUTPUT_DIR)

    # =========================================================================
    # STEP 6 — Análise de Sensibilidade / Stress Test
    # =========================================================================
    banner("STEP 6 — Stress Test (Sensibilidade)")
    sensitivity_df = ecl_sensitivity_analysis(portfolio, pd_shock=0.30, lgd_shock=0.15)

    base_ecl = portfolio["ecl_final"].sum()
    print(f"\n  ECL base: R$ {base_ecl:,.2f}")
    print(f"\n  Cenários de Estresse:")
    print(f"  {'Mult. PD':>10} {'Mult. LGD':>11} {'ECL Stress':>18} "
          f"{'Δ ECL':>16} {'Δ %':>8}")
    print(f"  {'-'*65}")
    for _, row in sensitivity_df.iterrows():
        print(f"  {row['pd_multiplier']:>10.2f} {row['lgd_multiplier']:>11.2f} "
              f"R$ {row['ecl_stress']:>15,.2f} "
              f"R$ {row['delta_ecl']:>13,.2f} {row['delta_pct']:>7.1f}%")

    sensitivity_df.to_csv(
        os.path.join(OUTPUT_DIR, "stress_test.csv"), index=False
    )

    # =========================================================================
    # STEP 7 — Gráficos
    # =========================================================================
    banner("STEP 7 — Visualizações")
    generate_all_plots(
        df=df,
        pd_results=pd_results,
        lgd_results=lgd_results,
        ead_results=ead_results,
        portfolio=portfolio,
        sensitivity_df=sensitivity_df,
        plots_dir=os.path.join(OUTPUT_DIR, "plots"),
    )

    # =========================================================================
    # Salva métricas para o dashboard web
    # =========================================================================
    metrics_web = {
        "pd": {
            "logistic": {
                "auc":  round(float(pd_results["logistic"]["auc_test"]), 4),
                "gini": round(float(pd_results["logistic"]["gini"]),     4),
                "ks":   round(float(pd_results["logistic"]["ks"]),       4),
            },
            "gbm": {
                "auc":  round(float(pd_results["gbm"]["auc_test"]), 4),
                "gini": round(float(pd_results["gbm"]["gini"]),     4),
                "ks":   round(float(pd_results["gbm"]["ks"]),       4),
            },
        },
        "lgd": {
            "ridge": {
                "r2":      round(float(lgd_results["ridge"]["r2"]),      4),
                "mae":     round(float(lgd_results["ridge"]["mae"]),     4),
                "spearman":round(float(lgd_results["ridge"]["spearman"]),4),
            },
            "gbm": {
                "r2":      round(float(lgd_results["gbm"]["r2"]),      4),
                "mae":     round(float(lgd_results["gbm"]["mae"]),     4),
                "spearman":round(float(lgd_results["gbm"]["spearman"]),4),
            },
        },
        "ead": {
            "ridge": {
                "r2":      round(float(ead_results["ridge"]["r2"]),      4),
                "mae":     round(float(ead_results["ridge"]["mae"]),     4),
                "spearman":round(float(ead_results["ridge"]["spearman"]),4),
            },
            "gbm": {
                "r2":      round(float(ead_results["gbm"]["r2"]),      4),
                "mae":     round(float(ead_results["gbm"]["mae"]),     4),
                "spearman":round(float(ead_results["gbm"]["spearman"]),4),
            },
        },
    }
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w") as _f:
        json.dump(metrics_web, _f, indent=2)
    print(f"  Métricas salvas → {metrics_path}")

    # =========================================================================
    # Resumo Final
    # =========================================================================
    elapsed = time.time() - t0
    banner("CONCLUÍDO")
    print(f"""
  Tempo total de execução : {elapsed:.1f}s

  Arquivos gerados:
    data/raw/portfolio.csv              — carteira sintética
    data/processed/portfolio_final.csv  — carteira com ECL
    outputs/models/                     — modelos (pd, lgd, ead)
    outputs/plots/                      — 10 gráficos de validação
    outputs/ecl_report_stages.csv       — ECL por stage
    outputs/stress_test.csv             — análise de sensibilidade

  Próximos passos sugeridos:
    - Aplicar backtesting com janelas temporais
    - Implementar curvas de sobrevivência (PD marginal por ano)
    - Modelar PiT vs TtC (Point-in-Time vs Through-the-Cycle)
    - Incorporar ajustes macroeconômicos (GDP, taxa de juros)
    - Validar modelos conforme SR 11-7 / ECB guidelines
    """)


if __name__ == "__main__":
    main()
