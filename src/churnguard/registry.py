"""Champion / challenger model registry with an automated quality gate.

A freshly trained model is a *challenger*. It is promoted to *champion* (the model
the API serves) only if it clears absolute quality bars AND does not regress
against the current champion. Every decision is appended to an audit log.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from churnguard.config import load_params, resolve

log = logging.getLogger("churnguard.registry")


@dataclass
class GateResult:
    promoted: bool
    challenger: str
    champion: str | None
    checks: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{'PROMOTED' if self.promoted else 'REJECTED'}: {self.challenger} (champion: {self.champion})"]
        lines += [f"  [{'x' if c['passed'] else ' '}] {c['name']}: {c['detail']}" for c in self.checks]
        return "\n".join(lines)


def _read_meta(path: Path) -> dict | None:
    f = path / "metadata.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def quality_gate(challenger: dict, champion: dict | None, rules: dict) -> GateResult:
    t = challenger["metrics"]["test"]
    checks = [
        {"name": "min_roc_auc", "passed": t["roc_auc"] >= rules["min_roc_auc"],
         "detail": f"{t['roc_auc']:.4f} >= {rules['min_roc_auc']}"},
        {"name": "min_pr_auc", "passed": t["pr_auc"] >= rules["min_pr_auc"],
         "detail": f"{t['pr_auc']:.4f} >= {rules['min_pr_auc']}"},
        {"name": "max_brier", "passed": t["brier"] <= rules["max_brier"],
         "detail": f"{t['brier']:.4f} <= {rules['max_brier']}"},
    ]
    if champion:
        # Prefer the champion re-scored on the challenger's holdout (apples to apples);
        # fall back to the champion's own historical test metrics.
        same = challenger["metrics"].get("champion_on_same_test")
        c, basis = (same, "same holdout") if same else (champion["metrics"]["test"], "historical")
        floor = c["roc_auc"] - rules["max_auc_drop_vs_champion"]
        checks.append({"name": "no_regression_vs_champion", "passed": t["roc_auc"] >= floor,
                       "detail": f"{t['roc_auc']:.4f} >= {floor:.4f} (champion {c['roc_auc']:.4f}, {basis})"})
        if same and "net_value" in same:
            nv = challenger["business"]["net_value_test"]
            checks.append({"name": "business_value_vs_champion", "passed": nv >= same["net_value"] - 0.02 * abs(same["net_value"]),
                           "detail": f"${nv:,.0f} vs champion ${same['net_value']:,.0f} (2% tolerance)"})
    return GateResult(all(c["passed"] for c in checks), challenger["version"],
                      champion["version"] if champion else None, checks)


def promote(params: dict | None = None, force: bool = False) -> GateResult:
    params = params or load_params()
    models = resolve(params["paths"]["models_dir"])
    chall_dir, champ_dir = models / "challenger", models / "champion"
    challenger = _read_meta(chall_dir)
    if challenger is None:
        raise FileNotFoundError("No challenger model found — run `churnguard train` first.")
    champion = _read_meta(champ_dir)

    result = quality_gate(challenger, champion, params["quality_gate"])
    if force and not result.promoted:
        result.checks.append({"name": "manual_override", "passed": True, "detail": "--force"})
        result.promoted = True

    if result.promoted:
        if champ_dir.exists():
            archive = models / "archive" / champion["version"]
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copytree(champ_dir, archive)
            shutil.rmtree(champ_dir)
        shutil.copytree(chall_dir, champ_dir)
        _set_mlflow_alias(params, challenger)

    entry = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **asdict(result)}
    history = models / "registry_log.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.info("\n%s", result.summary())
    return result


def _set_mlflow_alias(params: dict, metadata: dict) -> None:
    try:
        import mlflow

        from churnguard.train import setup_mlflow

        setup_mlflow(params)
        client = mlflow.MlflowClient()
        name = params["train"]["registered_model_name"]
        versions = client.search_model_versions(f"run_id = '{metadata['mlflow_run_id']}'")
        if versions:
            client.set_registered_model_alias(name, "champion", versions[0].version)
    except Exception as exc:
        log.warning("Could not set MLflow alias: %s", exc)
