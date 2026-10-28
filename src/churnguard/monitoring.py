"""Data & prediction drift monitoring.

Compares a production batch against the training reference with:
  * PSI (Population Stability Index) for every feature, numeric and categorical
  * Kolmogorov–Smirnov test for numeric features
  * PSI on the model's score distribution (prediction drift)

Outputs a JSON decision (`retrain_required`) consumed by the continuous-training
workflow, plus a self-contained HTML report for humans.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churnguard.config import load_params, resolve
from churnguard.data.validate import validate_or_raise
from churnguard.features import BASE_NUMERIC, CATEGORICAL

log = logging.getLogger("churnguard.monitoring")
EPS = 1e-4


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to value frequencies
        return psi_categorical(pd.Series(ref).round(6), pd.Series(cur).round(6))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(psi: float, cfg: dict) -> str:
    return "alert" if psi >= cfg["psi_alert"] else "warning" if psi >= cfg["psi_warning"] else "ok"


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict, model_bundle: dict | None = None) -> dict:
    features = []
    for col in BASE_NUMERIC:
        r, c = reference[col].to_numpy(float), current[col].to_numpy(float)
        psi = psi_numeric(r, c)
        ks = ks_2samp(r[~np.isnan(r)], c[~np.isnan(c)])
        features.append({
            "feature": col, "type": "numeric", "psi": round(psi, 4),
            "ks_stat": round(float(ks.statistic), 4), "ks_pvalue": float(ks.pvalue),
            "ref_mean": round(float(np.nanmean(r)), 3), "cur_mean": round(float(np.nanmean(c)), 3),
            "status": _status(psi, cfg),
        })
    for col in CATEGORICAL:
        psi = psi_categorical(reference[col], current[col])
        top_ref = reference[col].value_counts(normalize=True).idxmax()
        features.append({
            "feature": col, "type": "categorical", "psi": round(psi, 4),
            "ref_mean": round(float((reference[col] == top_ref).mean()), 3),
            "cur_mean": round(float((current[col] == top_ref).mean()), 3),
            "detail": f"share of '{top_ref}'", "status": _status(psi, cfg),
        })

    drifted = [f for f in features if f["status"] == "alert"]
    share = len(drifted) / len(features)

    prediction = None
    if model_bundle is not None:
        model = model_bundle["model"]
        s_ref = model.predict_proba(reference)[:, 1]
        s_cur = model.predict_proba(current)[:, 1]
        t = model_bundle["threshold"]
        prediction = {
            "score_psi": round(psi_numeric(s_ref, s_cur), 4),
            "ref_mean_score": round(float(s_ref.mean()), 4),
            "cur_mean_score": round(float(s_cur.mean()), 4),
            "ref_flag_rate": round(float((s_ref >= t).mean()), 4),
            "cur_flag_rate": round(float((s_cur >= t).mean()), 4),
        }

    reasons = []
    if share > cfg["max_drifted_feature_share"]:
        reasons.append(f"{len(drifted)}/{len(features)} features in PSI alert ({share:.0%} > "
                       f"{cfg['max_drifted_feature_share']:.0%})")
    if prediction and prediction["score_psi"] >= cfg["score_psi_alert"]:
        reasons.append(f"prediction drift: score PSI {prediction['score_psi']:.3f} >= {cfg['score_psi_alert']}")

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_reference": len(reference),
        "n_current": len(current),
        "drifted_features": [f["feature"] for f in drifted],
        "drifted_share": round(share, 4),
        "prediction_drift": prediction,
        "retrain_required": bool(reasons),
        "reasons": reasons,
        "features": sorted(features, key=lambda f: -f["psi"]),
    }


def write_html(report: dict, out: Path, cfg: dict) -> Path:
    max_psi = max([f["psi"] for f in report["features"]] + [cfg["psi_alert"] * 1.5])
    color = {"ok": "#16a34a", "warning": "#d97706", "alert": "#dc2626"}
    rows = []
    for f in report["features"]:
        w = min(100, 100 * f["psi"] / max_psi)
        rows.append(
            f"<tr><td><b>{html.escape(f['feature'])}</b><br><small>{f['type']}</small></td>"
            f"<td><div class='bar'><div style='width:{w:.1f}%;background:{color[f['status']]}'></div></div></td>"
            f"<td>{f['psi']:.3f}</td><td>{f['ref_mean']}</td><td>{f['cur_mean']}</td>"
            f"<td><span class='pill' style='background:{color[f['status']]}'>{f['status']}</span></td></tr>"
        )
    pred = report["prediction_drift"] or {}
    verdict = "RETRAIN REQUIRED" if report["retrain_required"] else "STABLE"
    vcolor = "#dc2626" if report["retrain_required"] else "#16a34a"
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in report["reasons"]) or "<li>No alert thresholds crossed.</li>"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ChurnGuard Drift Report</title>
<style>
body{{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:#f8fafc;color:#1f2933}}
main{{max-width:1000px;margin:0 auto;padding:32px 16px}} h1{{margin:0 0 4px}} .sub{{color:#616e7c;margin:0 0 20px}}
.verdict{{display:inline-block;color:#fff;background:{vcolor};padding:6px 14px;border-radius:999px;font-weight:700}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:#fff;border:1px solid #e4e7eb;border-radius:12px;padding:14px}} .kpi b{{display:block;font-size:22px}}
.kpi span{{color:#616e7c;font-size:13px}} section{{background:#fff;border:1px solid #e4e7eb;border-radius:12px;padding:18px;margin-bottom:16px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th{{text-align:left;color:#616e7c;font-weight:600}}
td,th{{padding:8px 6px;border-bottom:1px solid #eef0f3;vertical-align:middle}}
.bar{{background:#eef0f3;border-radius:6px;height:10px;min-width:120px}} .bar div{{height:10px;border-radius:6px}}
.pill{{color:#fff;border-radius:999px;padding:2px 10px;font-size:12px}}
</style></head><body><main>
<h1>ChurnGuard — Drift Report</h1>
<p class="sub">{report['n_reference']:,} reference vs {report['n_current']:,} production rows · {report['generated_at']}</p>
<span class="verdict">{verdict}</span>
<div class="kpis">
<div class="kpi"><span>Features in alert</span><b>{len(report['drifted_features'])} / {len(report['features'])}</b></div>
<div class="kpi"><span>Score PSI</span><b>{pred.get('score_psi', 'n/a')}</b></div>
<div class="kpi"><span>Mean churn score</span><b>{pred.get('ref_mean_score', '–')} → {pred.get('cur_mean_score', '–')}</b></div>
<div class="kpi"><span>Customers flagged</span><b>{pred.get('ref_flag_rate', 0):.1%} → {pred.get('cur_flag_rate', 0):.1%}</b></div>
</div>
<section><h2>Decision</h2><ul>{reasons}</ul>
<p><small>PSI ≥ {cfg['psi_warning']} warning, ≥ {cfg['psi_alert']} alert. Retrain when more than
{cfg['max_drifted_feature_share']:.0%} of features alert or score PSI ≥ {cfg['score_psi_alert']}.</small></p></section>
<section><h2>Feature stability</h2><table><tr><th>Feature</th><th>PSI</th><th></th><th>Reference</th><th>Current</th><th>Status</th></tr>
{''.join(rows)}</table><p><small>Reference/Current show the mean (numeric) or the share of the most common category.</small></p></section>
</main></body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out


def monitor(params: dict | None = None, current_path: str | Path | None = None) -> dict:
    params = params or load_params()
    cfg = params["monitoring"]
    champion = resolve(params["paths"]["models_dir"]) / "champion"
    if not (champion / "reference.csv.gz").exists():
        raise FileNotFoundError("No champion reference snapshot — train and promote a model first.")
    reference = pd.read_csv(champion / "reference.csv.gz")
    current = pd.read_csv(current_path or resolve(params["data"]["current_path"]))
    validate_or_raise(current, require_target=False)  # never compute drift on broken data
    bundle = joblib.load(champion / "model.joblib")
    report = detect_drift(reference, current, cfg, bundle)
    reports = resolve(params["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, reports / "drift_report.html", cfg)
    log.info("Drift check: %s | drifted=%s | reasons=%s",
             "RETRAIN" if report["retrain_required"] else "stable", report["drifted_features"], report["reasons"])
    return report
