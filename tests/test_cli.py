import json

from churnguard.data.generate import generate_customers
from churnguard.pipeline import main


def test_cli_validate_pass_and_fail(project, tmp_path, capsys):
    good = tmp_path / "good.csv"
    generate_customers(300, seed=1).to_csv(good, index=False)
    assert main(["validate", str(good)]) == 0

    bad = tmp_path / "bad.csv"
    df = generate_customers(300, seed=1)
    df.loc[:20, "tenure_months"] = -5
    df.to_csv(bad, index=False)
    assert main(["validate", str(bad)]) == 1
    assert "tenure_months" in capsys.readouterr().out


def test_cli_simulate_monitor_summary(project, capsys, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    outputs = tmp_path / "outputs.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    assert main(["simulate", "--drift", "--n-customers", "1500"]) == 0
    assert main(["monitor"]) == 0
    assert "retrain_required=" in outputs.read_text()
    assert main(["summary"]) == 0
    text = summary.read_text()
    assert "Quality gate" in text and "Drift" in text
    capsys.readouterr()


def test_cli_promote_is_idempotent(project):
    assert main(["promote"]) == 0
    log = (project["root"] / "models" / "registry_log.jsonl").read_text().strip().splitlines()
    assert json.loads(log[-1])["challenger"] == project["metadata"]["version"]
