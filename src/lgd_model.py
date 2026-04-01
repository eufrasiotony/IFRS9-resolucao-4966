"""
IFRS 9 — Modelo de LGD (Loss Given Default)
=============================================
Estima o percentual da exposição que NÃO será recuperado em caso de default.

    LGD = 1 − Taxa de Recuperação

Características do LGD:
  - Variável limitada ao intervalo [0, 1]
  - Geralmente bimodal (muitos defaults totais e muitas recuperações totais)
  - Fortemente influenciada pela presença e qualidade da garantia

Abordagem de modelagem (2 estágios):
  Estágio 1 — Regressão logística com transformação logit do LGD
              (lida bem com a natureza limitada da variável)
  Estágio 2 — Gradient Boosting para capturar não-linearidades

  O modelo final usa a média das predições como ensemble simples.

Avaliação:
  - R² (variância explicada)
  - MAE (erro absoluto médio)
  - Correlação de Spearman (ranking)
  - Distribuição prevista vs. real
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
    "ltv_ratio",
    "outstanding_balance",
    "loan_amount",
    "utilization_rate",
    "remaining_months",
    "interest_rate",
    "collateral_ratio",     # derivada
    "balance_log",          # derivada
]

CATEGORICAL_FEATURES = ["loan_type", "employment_type"]

TARGET = "lgd_true"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas para o modelo de LGD."""
    df = df.copy()
    # Razão colateral / saldo: quanto maior, menor o LGD esperado
    df["collateral_ratio"] = np.where(
        df["outstanding_balance"] > 0,
        df["collateral_value"] / df["outstanding_balance"],
        0.0,
    ).clip(0, 5)
    df["balance_log"] = np.log1p(df["outstanding_balance"])
    return df


# ---------------------------------------------------------------------------
# Transformação logit para variáveis limitadas [0, 1]
# ---------------------------------------------------------------------------
def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Transforma variável de [0,1] para (-∞, +∞)."""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Inversa do logit: converte de volta para [0, 1]."""
    return 1 / (1 + np.exp(-x))


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
        # Ridge regression na escala logit do LGD
        estimator = Ridge(alpha=1.0)
    elif model_type == "gbm":
        estimator = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
    else:
        raise ValueError(f"model_type deve ser 'ridge' ou 'gbm'.")

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def train_lgd_model(df: pd.DataFrame,
                    output_dir: str = "outputs") -> tuple[dict, pd.DataFrame]:
    """
    Treina modelos de LGD **apenas nos contratos em default**.

    Retorna
    -------
    results    : dict com métricas e modelos treinados
    lgd_scores : DataFrame com ['loan_id', 'lgd_pred'] para toda a carteira
    """
    print("\n" + "=" * 55)
    print("  MODELO DE LGD — Loss Given Default")
    print("=" * 55)

    # LGD é modelado apenas para contratos que já defaultaram
    df_def = df[df["default_flag"] == 1].copy()
    print(f"\n  Contratos em default disponíveis: {len(df_def):,}")
    print(f"  LGD médio (real): {df_def[TARGET].mean():.2%}")
    print(f"  LGD mediano (real): {df_def[TARGET].median():.2%}")

    df_feat = build_features(df_def)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # O modelo Ridge é treinado no espaço logit do LGD
    y_raw    = df_feat[TARGET].values
    y_logit  = logit(y_raw)

    # O GBM é treinado diretamente no LGD original
    X_tr_r, X_te_r, y_tr_logit, y_te_logit, y_tr_raw, y_te_raw = (
        *train_test_split(X, y_logit, y_raw, test_size=0.20,
                          random_state=42),
    )

    results: dict = {}

    # --- Ridge na escala logit ---
    pipe_ridge = _make_pipeline("ridge")
    pipe_ridge.fit(X_tr_r, y_tr_logit)
    pred_logit_te = pipe_ridge.predict(X_te_r)
    pred_ridge_te = sigmoid(pred_logit_te)   # volta para [0,1]

    r2_ridge  = r2_score(y_te_raw, pred_ridge_te)
    mae_ridge = mean_absolute_error(y_te_raw, pred_ridge_te)
    sp_ridge  = spearmanr(y_te_raw, pred_ridge_te).statistic

    print(f"\n  ── RIDGE (logit-transform) ──")
    print(f"     R²          : {r2_ridge:.4f}")
    print(f"     MAE         : {mae_ridge:.4f}")
    print(f"     Spearman r  : {sp_ridge:.4f}")

    results["ridge"] = {
        "pipeline":    pipe_ridge,
        "r2":          r2_ridge,
        "mae":         mae_ridge,
        "spearman":    sp_ridge,
        "pred_test":   pred_ridge_te,
        "y_test":      y_te_raw,
        "logit_mode":  True,
    }

    # --- GBM direto no LGD ---
    X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
        X, y_raw, test_size=0.20, random_state=42
    )
    pipe_gbm = _make_pipeline("gbm")
    pipe_gbm.fit(X_tr_g, y_tr_g)
    pred_gbm_te = pipe_gbm.predict(X_te_g).clip(0, 1)

    r2_gbm  = r2_score(y_te_g, pred_gbm_te)
    mae_gbm = mean_absolute_error(y_te_g, pred_gbm_te)
    sp_gbm  = spearmanr(y_te_g, pred_gbm_te).statistic

    print(f"\n  ── GBM ──")
    print(f"     R²          : {r2_gbm:.4f}")
    print(f"     MAE         : {mae_gbm:.4f}")
    print(f"     Spearman r  : {sp_gbm:.4f}")

    results["gbm"] = {
        "pipeline":   pipe_gbm,
        "r2":         r2_gbm,
        "mae":        mae_gbm,
        "spearman":   sp_gbm,
        "pred_test":  pred_gbm_te,
        "y_test":     y_te_g,
        "logit_mode": False,
    }

    # -----------------------------------------------------------------------
    # Salvar modelos
    # -----------------------------------------------------------------------
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    for name, res in results.items():
        joblib.dump(res["pipeline"],
                    os.path.join(output_dir, "models", f"lgd_{name}.pkl"))

    # -----------------------------------------------------------------------
    # LGD previsto para toda a carteira (usar GBM como melhor modelo)
    # Para contratos sem default as estimativas são usadas para calcular a ECL
    # -----------------------------------------------------------------------
    df_all_feat = build_features(df)
    X_all = df_all_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    best = results["gbm"]["pipeline"]
    lgd_pred = best.predict(X_all).clip(0.01, 0.99)

    lgd_scores = pd.DataFrame({
        "loan_id":  df["loan_id"].values,
        "lgd_pred": lgd_pred.round(4),
    })

    print(f"\n  LGD médio previsto (carteira): {lgd_pred.mean():.2%}")

    return results, lgd_scores
