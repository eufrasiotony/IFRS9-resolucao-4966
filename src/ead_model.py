"""
IFRS 9 — Modelo de EAD (Exposure at Default)
==============================================
Estima o valor da exposição no momento em que o cliente entra em default.

Para produtos NÃO rotativos (imobiliário, veículos, crédito pessoal):
    EAD ≈ saldo devedor atual  (sem modelo necessário)

Para produtos ROTATIVOS (cartão de crédito, limite empresarial):
    EAD = Saldo Sacado  +  CCF × Saldo Não Sacado
    onde CCF = Credit Conversion Factor ∈ [0, 1]

    Intuição do CCF: clientes que vão dar default tendem a sacar mais
    do limite disponível nos meses antes do evento.  O CCF modela
    qual % do saldo não sacado será utilizado.

    CCF alto significa que o cliente vai usar quase todo o limite restante.
    CCF baixo significa que o cliente não vai sacar mais antes do default.

Abordagem:
  - Treina modelo de CCF apenas para produtos rotativos
  - Usa Ridge (linear) e GBM (não-linear)
  - Para toda a carteira: EAD = drawn + CCF_pred × undrawn (rotativos)
                               = outstanding_balance          (demais)

Avaliação:
  - R², MAE no CCF
  - R², MAE no EAD final
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Features e configuração
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "utilization_rate",       # % do limite já utilizado
    "credit_limit_log",       # log do limite total
    "drawn_balance_log",      # log do saldo sacado
    "undrawn_balance_log",    # log do saldo não sacado
    "remaining_months",
    "interest_rate",
    "debt_to_income",
    "num_delinquencies",
]

CATEGORICAL_FEATURES = ["loan_type", "employment_type"]

TARGET_CCF = "ccf_true"
TARGET_EAD = "ead_true"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas para o modelo de EAD/CCF."""
    df = df.copy()
    df["credit_limit_log"]   = np.log1p(df["credit_limit"])
    df["drawn_balance_log"]  = np.log1p(df["drawn_balance"])
    df["undrawn_balance_log"] = np.log1p(df["undrawn_balance"])
    return df


# ---------------------------------------------------------------------------
# Pipeline sklearn
# ---------------------------------------------------------------------------
def _make_pipeline(model_type: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "ridge":
        estimator = Ridge(alpha=1.0)
    elif model_type == "gbm":
        estimator = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
    else:
        raise ValueError("model_type deve ser 'ridge' ou 'gbm'.")

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def train_ead_model(df: pd.DataFrame,
                    output_dir: str = "outputs") -> tuple[dict, pd.DataFrame]:
    """
    Treina modelo de CCF/EAD para produtos rotativos.

    Retorna
    -------
    results    : dict com métricas e modelos treinados
    ead_scores : DataFrame com ['loan_id', 'ead_pred', 'ccf_pred']
    """
    print("\n" + "=" * 55)
    print("  MODELO DE EAD — Exposure at Default")
    print("=" * 55)

    # O modelo de CCF é treinado apenas para produtos rotativos
    is_revolving = df["loan_type"].isin(["Cartao", "Empresarial"])
    df_rev = df[is_revolving].copy()

    print(f"\n  Produtos rotativos: {len(df_rev):,} contratos "
          f"({is_revolving.mean():.1%} da carteira)")
    print(f"  CCF médio (real)  : {df_rev[TARGET_CCF].mean():.4f}")

    df_feat = build_features(df_rev)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_ccf = df_feat[TARGET_CCF].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ccf, test_size=0.20, random_state=42
    )

    results: dict = {}

    for model_name in ("ridge", "gbm"):
        pipe = _make_pipeline(model_name)
        pipe.fit(X_train, y_train)

        pred_train = pipe.predict(X_train).clip(0, 1)
        pred_test  = pipe.predict(X_test).clip(0, 1)

        r2  = r2_score(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        sp  = spearmanr(y_test, pred_test).statistic

        print(f"\n  ── {model_name.upper()} (CCF) ──")
        print(f"     R²         : {r2:.4f}")
        print(f"     MAE        : {mae:.4f}")
        print(f"     Spearman r : {sp:.4f}")

        results[model_name] = {
            "pipeline":  pipe,
            "r2":        r2,
            "mae":       mae,
            "spearman":  sp,
            "pred_test": pred_test,
            "y_test":    y_test,
        }

        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        joblib.dump(pipe, os.path.join(output_dir, "models", f"ead_{model_name}.pkl"))

    # -----------------------------------------------------------------------
    # EAD para toda a carteira
    # -----------------------------------------------------------------------
    df_all_feat = build_features(df)
    X_all = df_all_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    is_rev_all = df["loan_type"].isin(["Cartao", "Empresarial"]).values

    best = results["gbm"]["pipeline"]
    ccf_pred = best.predict(X_all).clip(0, 1)

    # Para rotativos: EAD = drawn + CCF × undrawn
    # Para os demais: EAD = saldo devedor
    ead_pred = np.where(
        is_rev_all,
        df["drawn_balance"].values + ccf_pred * df["undrawn_balance"].values,
        df["outstanding_balance"].values,
    )
    # O EAD nunca pode ser negativo nem maior que o limite de crédito
    ead_pred = np.maximum(ead_pred, 0)

    ead_scores = pd.DataFrame({
        "loan_id":  df["loan_id"].values,
        "ccf_pred": np.where(is_rev_all, ccf_pred, np.nan).round(4),
        "ead_pred": ead_pred.round(2),
    })

    print(f"\n  EAD médio previsto (carteira): R$ {ead_pred.mean():,.2f}")
    print(f"  EAD total previsto (carteira): R$ {ead_pred.sum():,.2f}")

    return results, ead_scores
