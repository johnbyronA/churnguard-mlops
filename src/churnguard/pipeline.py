"""ChurnGuard command line — one entry point for every pipeline stage.

    churnguard generate                 # synthetic training data
    churnguard simulate --drift         # a new "production" batch (optionally drifted)
    churnguard validate data.csv        # data contract check
    churnguard train [--n-trials 30]    # Optuna + MLflow, writes challenger
    churnguard promote                  # quality gate -> champion
    churnguard monitor                  # drift report + retrain decision
    churnguard summary                  # markdown run summary (GitHub job page)
    churnguard run-all                  # generate -> train -> promote -> simulate -> monitor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from churnguard.config import load_params, resolve
from churnguard.data.generate import generate_customers
from churnguard.data.validate import validate


def _write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s rows -> %s", len(df), path)


def cmd_generate(args, params):
    n = args.n_customers or params["data"]["n_customers"]
    _write(generate_customers(n, seed=params["seed"]), resolve(params["data"]["raw_path"]))


def cmd_simulate(args, params):
    n = args.n_customers or params["data"]["n_customers"] // 4
    seed = args.seed if args.seed is not None else params["seed"] + 1
    labelled = generate_customers(n, seed=seed, drift=args.drift)
    _write(labelled.drop(columns=["churn"]), resolve(params["data"]["current_path"]))
    if args.label_feedback:
        # Ground truth arrives later in real life; here we append the labelled batch to the
        # training set so the continuous-training job can learn the new behaviour.
        raw_path = resolve(params["data"]["raw_path"])
        combined = pd.concat([pd.read_csv(raw_path), labelled], ignore_index=True)
        _write(combined, raw_path)


def cmd_summary(args, params):
    """Markdown summary for the GitHub Actions job page."""
    reports = resolve(params["paths"]["reports_dir"])
    models = resolve(params["paths"]["models_dir"])
    lines = ["## ChurnGuard pipeline run", ""]
    metrics_file = reports / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        t, b = m["test"], m["business"]
        lines += [f"**Challenger `{m['version']}`**", "", "| Metric | Value |", "|---|---|",
                  f"| ROC-AUC | {t['roc_auc']:.4f} |", f"| PR-AUC | {t['pr_auc']:.4f} |",
                  f"| Brier | {t['brier']:.4f} |", f"| Top-decile lift | {t['lift_top_decile']:.2f}x |",
                  f"| Decision threshold | {b['threshold']:.2f} |",
                  f"| Campaign net value (test) | ${b['net_value_test']:,.0f} |",
                  f"| ...at naive 0.50 threshold | ${b['net_value_at_0_5']:,.0f} |",
                  f"| ...contacting everyone | ${b['net_value_contact_all']:,.0f} |", ""]
    log_file = models / "registry_log.jsonl"
    if log_file.exists():
        last = json.loads(log_file.read_text().strip().splitlines()[-1])
        lines += [f"### Quality gate: {'✅ PROMOTED' if last['promoted'] else '❌ REJECTED'}", ""]
        lines += [f"- {'✅' if c['passed'] else '❌'} `{c['name']}` — {c['detail']}" for c in last["checks"]]
        lines.append("")
    drift_file = reports / "drift_report.json"
    if drift_file.exists():
        d = json.loads(drift_file.read_text())
        lines += [f"### Drift: {'⚠️ RETRAIN REQUIRED' if d['retrain_required'] else '✅ stable'}", ""]
        lines += [f"- {r}" for r in d["reasons"]] or ["- no alert thresholds crossed"]
        lines += ["", "| Feature | PSI | Status |", "|---|---|---|"]
        lines += [f"| {f['feature']} | {f['psi']:.3f} | {f['status']} |" for f in d["features"][:8]]
    text = "\n".join(lines) + "\n"
    target = os.getenv("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)


def cmd_validate(args, params):
    report = validate(pd.read_csv(args.path), require_target=not args.no_target)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


def cmd_train(args, params):
    from churnguard.train import train

    meta = train(params, n_trials=args.n_trials)
    print(json.dumps({"version": meta["version"], **meta["metrics"]["test"],
                      "net_value_test": meta["business"]["net_value_test"]}, indent=2))


def cmd_promote(args, params):
    from churnguard.registry import promote

    result = promote(params, force=args.force)
    _github_output("promoted", str(result.promoted).lower())
    return 0 if result.promoted or not args.strict else 1


def cmd_monitor(args, params):
    from churnguard.monitoring import monitor

    report = monitor(params, current_path=args.path)
    print(json.dumps({k: report[k] for k in ("retrain_required", "reasons", "drifted_features",
                                             "prediction_drift")}, indent=2))
    _github_output("retrain_required", str(report["retrain_required"]).lower())


def cmd_run_all(args, params):
    cmd_generate(args, params)
    cmd_train(args, params)
    args.force, args.strict = False, False
    cmd_promote(args, params)
    args.drift, args.seed, args.label_feedback = True, None, False
    args.n_customers = None
    cmd_simulate(args, params)
    args.path = None
    cmd_monitor(args, params)


def _github_output(key: str, value: str) -> None:
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(prog="churnguard", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--n-customers", type=int)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("simulate")
    p.add_argument("--n-customers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--label-feedback", action="store_true",
                   help="also append the labelled batch to the training data")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("summary")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("validate")
    p.add_argument("path")
    p.add_argument("--no-target", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("train")
    p.add_argument("--n-trials", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("promote")
    p.add_argument("--force", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 if the challenger is rejected")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("monitor")
    p.add_argument("--path")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("run-all")
    p.add_argument("--n-customers", type=int)
    p.add_argument("--n-trials", type=int)
    p.set_defaults(func=cmd_run_all)

    args = parser.parse_args(argv)
    return args.func(args, load_params()) or 0


if __name__ == "__main__":
    sys.exit(main())
