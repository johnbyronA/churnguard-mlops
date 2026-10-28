"""Figures and a self-contained HTML model report (model card)."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2563eb"
ACCENT_2 = "#f97316"
GRID = "#e4e7eb"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_roc_pr(y, proba, out: Path, auc: float, pr_auc: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y, proba)
    axes[0].plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"Model (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Random")
    _style(axes[0], "ROC curve", "False positive rate", "True positive rate")
    axes[0].legend(frameon=False)
    prec, rec, _ = precision_recall_curve(y, proba)
    axes[1].plot(rec, prec, color=ACCENT, lw=2.2, label=f"Model (AP {pr_auc:.3f})")
    axes[1].axhline(np.mean(y), color=MUTED, ls="--", lw=1, label=f"Base rate {np.mean(y):.2f}")
    _style(axes[1], "Precision–Recall curve", "Recall", "Precision")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_calibration(y, proba_raw, proba_cal, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for p, label, color in [(proba_raw, "Uncalibrated", MUTED), (proba_cal, "Calibrated", ACCENT)]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", lw=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], color=INK, ls=":", lw=1, label="Perfect")
    _style(ax, "Probability calibration", "Predicted probability", "Observed churn rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_value_curve(thresholds, values, best_t: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thresholds, values / 1000, color=ACCENT, lw=2.2)
    ax.axvline(best_t, color=ACCENT_2, ls="--", lw=1.5, label=f"Chosen threshold {best_t:.2f}")
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2, label="Naive 0.50")
    ax.axhline(0, color=INK, lw=0.8)
    _style(ax, "Retention campaign net value vs threshold", "Decision threshold", "Net value (USD thousands)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_importance(importance: pd.Series, out: Path, top: int = 12) -> Path:
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color=ACCENT)
    _style(ax, "What drives churn (mean |SHAP|)", "Mean absolute contribution (log-odds)", "")
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _img(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(Path(path).stem)}">'


def write_model_report(metadata: dict, figures: dict[str, Path], out: Path) -> Path:
    m = metadata["metrics"]["test"]
    biz = metadata["business"]
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else ""
        for k, v in m.items()
    )
    params = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                     for k, v in metadata["hyperparameters"].items())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChurnGuard Model Card</title>
<style>
:root{{--ink:#1f2933;--muted:#616e7c;--line:#e4e7eb;--bg:#f8fafc;--card:#fff;--accent:#2563eb}}
body{{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px}}
h1{{margin:0 0 4px;font-size:28px}} .sub{{color:var(--muted);margin:0 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.kpi b{{display:block;font-size:26px;margin-top:4px}} .kpi span{{color:var(--muted);font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:16px}}
h2{{font-size:17px;margin:0 0 12px}} img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td{{border-bottom:1px solid var(--line);padding:6px 4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
code{{background:#eef2ff;padding:1px 5px;border-radius:4px}}
</style></head><body><main>
<h1>ChurnGuard — Model Card</h1>
<p class="sub">Version <code>{html.escape(metadata['version'])}</code> · trained {html.escape(metadata['trained_at'])}
 · {metadata['data']['n_train']:,} train / {metadata['data']['n_test']:,} test customers · git <code>{html.escape(metadata['git_sha'])}</code></p>
<div class="kpis">
<div class="kpi"><span>ROC-AUC (test)</span><b>{m['roc_auc']:.3f}</b></div>
<div class="kpi"><span>PR-AUC (test)</span><b>{m['pr_auc']:.3f}</b></div>
<div class="kpi"><span>Brier score</span><b>{m['brier']:.3f}</b></div>
<div class="kpi"><span>Top-decile lift</span><b>{m['lift_top_decile']:.2f}×</b></div>
<div class="kpi"><span>Campaign value (test)</span><b>${biz['net_value_test']:,.0f}</b></div>
<div class="kpi"><span>vs. contact everyone</span><b>+${biz['net_value_test'] - biz['net_value_contact_all']:,.0f}</b></div>
</div>
<section><h2>Discrimination</h2>{_img(figures['roc_pr'])}</section>
<div class="grid">
<section><h2>Business-optimal threshold</h2>{_img(figures['value_curve'])}
<p>Threshold <b>{biz['threshold']:.2f}</b> maximises expected net value with offer cost ${biz['offer_cost']:.0f},
retention success {biz['retention_success_rate']:.0%} and a {biz['value_horizon_months']}-month horizon.</p></section>
<section><h2>Calibration</h2>{_img(figures['calibration'])}</section>
</div>
<section><h2>Explainability</h2>{_img(figures['importance'])}</section>
<div class="grid">
<section><h2>Test metrics</h2><table>{rows}</table></section>
<section><h2>Hyperparameters (Optuna best)</h2><table>{params}</table></section>
</div>
<section><h2>Intended use & limits</h2><p>Ranks active subscribers by churn risk to prioritise retention offers.
Trained on synthetic data that mimics a subscription business; not for decisions about real people.
Monitored for data and prediction drift; retrained automatically when drift exceeds the alert policy.</p></section>
</main></body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out
