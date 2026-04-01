"""
IFRS 9 — Modelo de PD (Probability of Default)
================================================
Estima a probabilidade de um cliente deixar de honrar suas obrigações.

Boas práticas regulatórias:
  - Estimativas Point-in-Time (PIT), não Through-the-Cycle
  - PD 12m → Stage 1
  - PD Lifetime → Stage 2 e 3 (usando a PD anual projetada)
  - Avaliação com AUC, Gini e KS

Modelos treinados:
  1. Regressão Logística (interpretável, regulatoriamente aceita)
  2. Gradient Boosting    (melhor poder preditivo)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "credit_score_current",
    "score_change",
    "debt_to_income",
    "ltv_ratio",
    "num_delinquencies",
    "days_past_due",
    "utilization_rate",
    "remaining_months",
    "employment_years",
    "income_log",          # derivada
]

CATEGORICAL_FEATURES = ["employment_type", "loan_type"]

TARGET = "default_flag"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas para o modelo de PD."""
    df = df.copy()
    df["income_log"] = np.log1p(df["annual_income"])
    return df


# ---------------------------------------------------------------------------
# Pipeline sklearn
# ---------------------------------------------------------------------------
def _make_pipeline(model_type: str) -> Pipeline:
    """Cria pipeline: pré-processamento + estimador."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "logistic":
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                       class_weight="balanced")
    elif model_type == "gbm":
        estimator = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
    else:
        raise ValueError(f"model_type deve ser 'logistic' ou 'gbm'.")

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


# ---------------------------------------------------------------------------
# Métricas de desempenho (crédito)
# ---------------------------------------------------------------------------
def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini = 2 × AUC − 1

    Um Gini > 0.40 é considerado bom para modelos de crédito.
    Gini representa a capacidade de separar bons de maus pagadores.
    """
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    KS (Kolmogorov-Smirnov): máxima distância entre as curvas cumulativas
    de bons e maus. Mede o poder discriminatório do score.

    KS > 0.30 é aceitável; > 0.45 é bom.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _print_metrics(name: str, auc_tr: float, auc_ts: float,
                   gini: float, ks: float, brier: float) -> None:
    print(f"\n  ── {name.upper()} ──")
    print(f"     AUC  (train): {auc_tr:.4f}")
    print(f"     AUC  (test) : {auc_ts:.4f}")
    print(f"     Gini (test) : {gini:.4f}   (bom se > 0.40)")
    print(f"     KS   (test) : {ks:.4f}   (bom se > 0.30)")
    print(f"     Brier Score : {brier:.4f}  (calibração, menor é melhor)")


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def train_pd_model(df: pd.DataFrame,
                   output_dir: str = "outputs") -> tuple[dict, pd.DataFrame]:
    """
    Treina modelos de PD, avalia e retorna os scores para toda a carteira.

    Retorna
    -------
    results   : dict com métricas e modelos treinados
    pd_scores : DataFrame com colunas ['loan_id', 'pd_12m', 'pd_lifetime']
    """
    print("\n" + "=" * 55)
    print("  MODELO DE PD — Probability of Default")
    print("=" * 55)

    df_feat = build_features(df)
    X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_feat[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n  Amostras: treino={len(X_train):,}  teste={len(X_test):,}")
    print(f"  Taxa de default: treino={y_train.mean():.2%}  teste={y_test.mean():.2%}")

    results: dict = {}

    for model_name in ("logistic", "gbm"):
        pipe = _make_pipeline(model_name)
        pipe.fit(X_train, y_train)

        prob_train = pipe.predict_proba(X_train)[:, 1]
        prob_test  = pipe.predict_proba(X_test)[:, 1]

        auc_tr  = roc_auc_score(y_train, prob_train)
        auc_ts  = roc_auc_score(y_test,  prob_test)
        gini    = gini_coefficient(y_test, prob_test)
        ks      = ks_statistic(y_test, prob_test)
        brier   = brier_score_loss(y_test, prob_test)

        _print_metrics(model_name, auc_tr, auc_ts, gini, ks, brier)

        results[model_name] = {
            "pipeline":   pipe,
            "auc_train":  auc_tr,
            "auc_test":   auc_ts,
            "gini":       gini,
            "ks":         ks,
            "brier":      brier,
            "y_test":     y_test.values,
            "prob_test":  prob_test,
            "fpr":        roc_curve(y_test, prob_test)[0],
            "tpr":        roc_curve(y_test, prob_test)[1],
        }

        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        joblib.dump(pipe, os.path.join(output_dir, "models", f"pd_{model_name}.pkl"))

    # -----------------------------------------------------------------------
    # Scores de PD para toda a carteira — usar melhor modelo (GBM)
    # -----------------------------------------------------------------------
    best = results["gbm"]["pipeline"]
    pd_12m = best.predict_proba(X)[:, 1]

    # PD Lifetime: assume taxa de default anual constante
    # PD_lifetime = 1 − (1 − PD_anual)^T   onde T = meses_restantes / 12
    pd_lifetime = compute_pd_lifetime(pd_12m, df_feat["remaining_months"].values)

    pd_scores = pd.DataFrame({
        "loan_id":     df_feat["loan_id"].values,
        "pd_12m":      pd_12m.round(6),
        "pd_lifetime": pd_lifetime.round(6),
    })

    print(f"\n  PD 12m médio da carteira  : {pd_12m.mean():.4f}")
    print(f"  PD Lifetime médio         : {pd_lifetime.mean():.4f}")

    return results, pd_scores


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def compute_pd_lifetime(pd_12m: np.ndarray,
                        remaining_months: np.ndarray) -> np.ndarray:
    """
    Converte PD 12m em PD Lifetime assumindo taxa de default constante.

    Fórmula:  PD_lifetime = 1 − (1 − pd_anual)^T
    onde T = remaining_months / 12

    Observação: em produção usa-se PD marginal por ano com curvas
    de sobrevivência.  Aqui usamos a simplificação para fins didáticos.
    """
    T = np.maximum(remaining_months, 1) / 12.0
    return (1 - (1 - pd_12m) ** T).clip(0, 1)


def load_pd_model(model_type: str = "gbm",
                  output_dir: str = "outputs") -> Pipeline:
    """Carrega modelo de PD salvo em disco."""
    path = os.path.join(output_dir, "models", f"pd_{model_type}.pkl")
    return joblib.load(path)
