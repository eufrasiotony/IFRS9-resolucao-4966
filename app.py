"""
IFRS 9 Course — Flask Web Application
======================================
Serve o site do curso de IFRS 9 com os resultados dos modelos.

Rotas:
  GET /                → Página inicial / Apresentação do curso
  GET /modulo/<n>      → Módulos 1–7
  GET /dashboard       → Dashboard final com KPIs e gráficos
  GET /plots/<file>    → Serve PNG de outputs/plots/
  GET /api/data        → JSON com dados do dashboard
"""

import json
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory

# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load() -> dict:
    data: dict = {}

    # ── Portfolio (portfolio_final.csv) ─────────────────────────────────────
    portfolio_path = os.path.join(BASE_DIR, "data", "processed", "portfolio_final.csv")
    if os.path.exists(portfolio_path):
        df = pd.read_csv(portfolio_path)

        stage_agg = (
            df.groupby("stage")
            .agg(n=("loan_id", "count"),
                 ead=("ead_pred", "sum"),
                 ecl=("ecl_final", "sum"),
                 pd_avg=("pd_12m", "mean"),
                 lgd_avg=("lgd_pred", "mean"))
            .reset_index()
        )
        stage_agg["cov"] = (stage_agg["ecl"] / stage_agg["ead"] * 100).round(2)

        prod_agg = (
            df.groupby("loan_type")
            .agg(n=("loan_id", "count"),
                 ead=("ead_pred", "sum"),
                 ecl=("ecl_final", "sum"),
                 dr=("default_flag", "mean"))
            .reset_index()
        )
        prod_agg["cov"] = (prod_agg["ecl"] / prod_agg["ead"] * 100).round(2)

        data["kpis"] = {
            "n_loans":      int(len(df)),
            "ead_M":        round(df["ead_pred"].sum() / 1e6, 1),
            "ecl_M":        round(df["ecl_final"].sum() / 1e6, 1),
            "coverage":     round(df["ecl_final"].sum() / df["ead_pred"].sum() * 100, 2),
            "default_rate": round(df["default_flag"].mean() * 100, 2),
            "avg_pd":       round(df["pd_12m"].mean() * 100, 2),
            "avg_lgd":      round(df["lgd_pred"].mean() * 100, 2),
            "stage1_pct":   round((df["stage"] == 1).mean() * 100, 1),
            "stage2_pct":   round((df["stage"] == 2).mean() * 100, 1),
            "stage3_pct":   round((df["stage"] == 3).mean() * 100, 1),
        }
        data["stage_data"] = stage_agg.to_dict("records")
        data["prod_data"]  = prod_agg.to_dict("records")

    # ── Stress test ──────────────────────────────────────────────────────────
    stress_path = os.path.join(BASE_DIR, "outputs", "stress_test.csv")
    if os.path.exists(stress_path):
        data["stress_data"] = pd.read_csv(stress_path).to_dict("records")
    else:
        data["stress_data"] = []

    # ── Model metrics ────────────────────────────────────────────────────────
    metrics_path = os.path.join(BASE_DIR, "outputs", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    else:
        # Fallback com valores do último run
        data["metrics"] = {
            "pd":  {"logistic": {"auc": 0.868, "gini": 0.736, "ks": 0.581},
                    "gbm":      {"auc": 0.857, "gini": 0.713, "ks": 0.572}},
            "lgd": {"ridge":    {"r2": 0.936, "mae": 0.055, "spearman": 0.844},
                    "gbm":      {"r2": 0.928, "mae": 0.058, "spearman": 0.826}},
            "ead": {"ridge":    {"r2": 0.034, "mae": 0.056, "spearman": 0.230},
                    "gbm":      {"r2": -0.014, "mae": 0.058, "spearman": 0.183}},
        }

    data["data_available"] = "kpis" in data

    # ── Credit portfolio KPIs (faixa de atraso × mês referência) ─────────────
    if data["data_available"]:
        df_fin  = pd.read_csv(portfolio_path)
        raw_path = os.path.join(BASE_DIR, "data", "raw", "portfolio.csv")
        if os.path.exists(raw_path):
            df_raw  = pd.read_csv(raw_path)[["loan_id", "days_past_due"]]
            df_full = df_fin.merge(df_raw, on="loan_id", how="left")
        else:
            df_full = df_fin.copy()
            df_full["days_past_due"] = 0
        data["port_kpis"] = _compute_port_kpis(df_full)
    else:
        data["port_kpis"] = {}

    return data


# ---------------------------------------------------------------------------
# DPD helpers
# ---------------------------------------------------------------------------
DPD_BINS   = [-1, 0, 30, 60, 90, 120, 99999]
DPD_LABELS = ["A00", "A01-30", "A31-60", "A61-90", "A91-120", "A120+"]

# Forward Markov transition matrix  P[i][j] = P(bucket_j | was_in_bucket_i)
# Calibrated on Brazilian retail credit industry benchmarks
_P = np.array([
    [0.920, 0.068, 0.008, 0.003, 0.001, 0.000],  # A00
    [0.500, 0.220, 0.180, 0.070, 0.020, 0.010],  # A01-30
    [0.150, 0.250, 0.280, 0.220, 0.070, 0.030],  # A31-60
    [0.050, 0.100, 0.200, 0.300, 0.250, 0.100],  # A61-90
    [0.030, 0.050, 0.080, 0.140, 0.300, 0.400],  # A91-120
    [0.020, 0.030, 0.050, 0.070, 0.130, 0.700],  # A120+
], dtype=float)

# Backward transition: P_back[j][i] = P(was_in_i | now_in_j), col-normalized P^T
_P_T     = _P.T
_P_back  = _P_T / _P_T.sum(axis=0, keepdims=True)


def _assign_dpd_bucket(dpd_series: "pd.Series") -> "pd.Series":
    import pandas as _pd
    return _pd.cut(dpd_series, bins=DPD_BINS, labels=DPD_LABELS, right=True)


def _simulate_ref_months(df: "pd.DataFrame", n_months: int = 6) -> list:
    """
    Simula n_months meses de referência usando cadeia de Markov reversa.
    Mês n_months-1 = dados reais; meses anteriores = simulados regressivamente.
    Retorna lista de dicts: {month_label, month_idx, dpd_bucket, n, ead, ecl,
                              pct_n, pct_ead, ead_M, ecl_M}.
    """
    rng = np.random.default_rng(seed=42)

    # bucket index per contract at current month
    bucket_series = _assign_dpd_bucket(df["days_past_due"].fillna(0))
    bucket_map    = {b: i for i, b in enumerate(DPD_LABELS)}
    cur_idx       = bucket_series.map(lambda x: bucket_map.get(str(x), 0)).to_numpy()

    ead_arr = df["ead_pred"].to_numpy()
    ecl_arr = df["ecl_final"].to_numpy()
    n_loans = len(df)

    # Simulate backward: snapshots[0] = oldest month, snapshots[n_months-1] = current
    snapshots = [None] * n_months
    snapshots[n_months - 1] = cur_idx.copy()

    for k in range(n_months - 2, -1, -1):
        prev = np.empty(n_loans, dtype=int)
        cur  = snapshots[k + 1]
        for i in range(n_loans):
            b    = cur[i]
            prob = _P_back[:, b]
            prev[i] = rng.choice(len(DPD_LABELS), p=prob)
        snapshots[k] = prev

    # Reference month labels (current = Mar/2026, going back)
    from datetime import date
    base  = date(2026, 3, 1)
    months_pt = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    month_labels = []
    for k in range(n_months):
        offset = n_months - 1 - k          # 0 = current
        m = base.month - offset
        y = base.year
        while m <= 0:
            m += 12; y -= 1
        month_labels.append(f"{months_pt[m - 1]}/{y}")

    records = []
    for k in range(n_months):
        idxs      = snapshots[k]
        total_n   = n_loans
        total_ead = ead_arr.sum()
        for bi, bname in enumerate(DPD_LABELS):
            mask = idxs == bi
            rec_n   = int(mask.sum())
            rec_ead = float(ead_arr[mask].sum())
            rec_ecl = float(ecl_arr[mask].sum())
            records.append({
                "month_label": month_labels[k],
                "month_idx":   k,
                "dpd_bucket":  bname,
                "n":           rec_n,
                "ead":         rec_ead,
                "ecl":         rec_ecl,
                "ead_M":       round(rec_ead / 1e6, 2),
                "ecl_M":       round(rec_ecl / 1e6, 2),
                "pct_n":       round(rec_n   / total_n   * 100, 1) if total_n   else 0.0,
                "pct_ead":     round(rec_ead / total_ead * 100, 1) if total_ead else 0.0,
            })
    return records


def _compute_port_kpis(df: "pd.DataFrame") -> dict:
    """Calcula KPIs de carteira: faixa de atraso × mês referência, roll rate, eficiência."""
    k: dict = {}
    df = df.copy()

    # ── DPD bucket atual ─────────────────────────────────────────────────────
    df["dpd_bucket"] = _assign_dpd_bucket(df["days_past_due"].fillna(0)).astype(str)

    # ── isDelayed (faixa > A00) ──────────────────────────────────────────────
    k["is_delayed_pct"] = round((df["dpd_bucket"] != "A00").mean() * 100, 2)
    k["is_delayed_n"]   = int((df["dpd_bucket"] != "A00").sum())
    k["default_rate"]   = round(df["default_flag"].mean() * 100, 2)

    # ── Distribuição atual por faixa + produto ───────────────────────────────
    faixa_agg = (
        df.groupby("dpd_bucket")
          .agg(n=("loan_id","count"), ead=("ead_pred","sum"), ecl=("ecl_final","sum"),
               defaulted=("default_flag","sum"))
          .reindex(DPD_LABELS).fillna(0).reset_index()
    )
    faixa_agg.rename(columns={"index": "dpd_bucket"}, inplace=True)
    total_n   = faixa_agg["n"].sum()
    total_ead = faixa_agg["ead"].sum()
    faixa_agg["pct_n"]   = (faixa_agg["n"]   / total_n   * 100).round(1)
    faixa_agg["pct_ead"] = (faixa_agg["ead"] / total_ead * 100).round(1)
    faixa_agg["dr_pct"]  = (faixa_agg["defaulted"] / faixa_agg["n"].replace(0, np.nan) * 100).fillna(0).round(1)
    faixa_agg["cov"]     = (faixa_agg["ecl"] / faixa_agg["ead"].replace(0, np.nan) * 100).fillna(0).round(2)
    faixa_agg["ead_M"]   = (faixa_agg["ead"] / 1e6).round(2)
    faixa_agg["ecl_M"]   = (faixa_agg["ecl"] / 1e6).round(2)
    k["faixa_atraso"] = faixa_agg.to_dict("records")

    # ── IsDelayed por produto ────────────────────────────────────────────────
    delayed_prod = (
        df.groupby("loan_type")
          .agg(n=("loan_id","count"),
               delayed=("dpd_bucket", lambda x: (x != "A00").sum()),
               defaulted=("default_flag","sum"),
               ead=("ead_pred","sum"))
          .reset_index()
    )
    delayed_prod["delayed_pct"] = (delayed_prod["delayed"]  / delayed_prod["n"] * 100).round(1)
    delayed_prod["default_pct"] = (delayed_prod["defaulted"]/ delayed_prod["n"] * 100).round(1)
    k["delayed_prod"] = delayed_prod.to_dict("records")

    # ── Faixa de atraso × mês referência (simulação Markov reversa) ───────────
    roll_months = _simulate_ref_months(df, n_months=6)
    k["roll_months"] = roll_months

    # Ordered month labels for template
    k["month_labels"] = list(dict.fromkeys(r["month_label"] for r in roll_months))

    # ── Roll Rate Matrix (faixa t-1 → faixa t, entre últimos 2 meses) ────────
    # Derive from P matrix (last two simulated months)
    sim_records = {r["dpd_bucket"]: {} for r in roll_months}
    last_idx  = max(r["month_idx"] for r in roll_months)
    cur_dist  = {r["dpd_bucket"]: r["pct_n"]
                 for r in roll_months if r["month_idx"] == last_idx}
    prev_dist = {r["dpd_bucket"]: r["pct_n"]
                 for r in roll_months if r["month_idx"] == last_idx - 1}

    roll_matrix = []
    for i, b_from in enumerate(DPD_LABELS):
        row = {"bucket_from": b_from, "prev_pct": round(prev_dist.get(b_from, 0), 1)}
        for j, b_to in enumerate(DPD_LABELS):
            row[b_to] = round(_P[i][j] * 100, 1)
        roll_matrix.append(row)
    k["roll_matrix"] = roll_matrix
    k["dpd_labels"]  = DPD_LABELS

    # ── Eficiência Encadeada e Diagonal (derivada da matriz P) ───────────────
    k["efic_diagonal"] = [
        {"bucket": b, "pct": round(_P[i][i] * 100, 1)}
        for i, b in enumerate(DPD_LABELS)
    ]

    flow_rate    = round((1 - _P[0][0]) * 100, 1)   # S1→S2 proxy: % que sai de A00
    cure_rate    = round(_P[1][0] * 100, 1)          # % que cura de A01-30 → A00
    default_flow = round(_P[3][4] * 100 + _P[3][5] * 100, 1)  # A61-90 → A91+
    k["efic_encadeada"] = {
        "flow_rate":    flow_rate,     # % saindo de A00 por mês
        "cure_rate":    cure_rate,     # % curando de A01-30 → A00
        "default_flow": default_flow,  # % de A61-90 indo para default
        "chain_pct":   round(flow_rate / 100 * (1 - cure_rate / 100) * 100, 2),
    }

    # ── Concentração HHI ─────────────────────────────────────────────────────
    prod_ead = df.groupby("loan_type")["ead_pred"].sum()
    shares   = (prod_ead / prod_ead.sum()).values
    k["hhi"] = round(float((shares ** 2).sum() * 10000), 0)

    return k


DATA = _load()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
MODULE_TEMPLATES = {
    1: "m1.html",
    2: "m2.html",
    3: "m3.html",
    4: "m4.html",
    5: "m5.html",
    6: "m6.html",
    7: "m7.html",
    8: "m8.html",
}


@app.route("/")
def index():
    return render_template("index.html", data=DATA, current="home")


@app.route("/modulo/<int:num>")
def modulo(num: int):
    tmpl = MODULE_TEMPLATES.get(num)
    if not tmpl:
        return "Módulo não encontrado", 404
    return render_template(tmpl, data=DATA, current=f"m{num}",
                           prev_url=f"/modulo/{num-1}" if num > 1 else "/",
                           next_url=f"/modulo/{num+1}" if num < 8 else "/dashboard")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", data=DATA, current="dashboard")


@app.route("/kpis")
def kpis():
    return render_template("kpis.html", data=DATA, current="kpis")


@app.route("/plots/<path:filename>")
def plots(filename: str):
    return send_from_directory(PLOTS_DIR, filename)


@app.route("/api/data")
def api_data():
    return jsonify(DATA)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
