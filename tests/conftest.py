import shutil
from pathlib import Path

import pytest

from churnguard.config import load_params
from churnguard.data.generate import generate_customers

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """An isolated project root with a tiny dataset and one trained + promoted model."""
    root = tmp_path_factory.mktemp("churnguard")
    shutil.copy(REPO / "params.yaml", root / "params.yaml")
    mp = pytest.MonkeyPatch()
    mp.setenv("CHURNGUARD_ROOT", str(root))
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root / 'mlflow.db'}")
    mp.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mp.chdir(root)

    params = load_params(root / "params.yaml")
    raw = root / params["data"]["raw_path"]
    raw.parent.mkdir(parents=True)
    generate_customers(4000, seed=7).to_csv(raw, index=False)

    from churnguard.registry import promote
    from churnguard.train import train

    metadata = train(params, n_trials=2)
    promote(params, force=True)
    yield {"root": root, "params": params, "metadata": metadata}
    mp.undo()


@pytest.fixture
def customers():
    return generate_customers(1500, seed=3)
