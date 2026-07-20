from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_clean_python(source: str) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).parents[1]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    source_root = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_root if not existing else source_root + os.pathsep + existing
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_classical_cli_and_artifact_metadata_do_not_import_torch() -> None:
    completed = _run_clean_python(
        "import sys\n"
        "import vpncat.cli\n"
        "import vpncat.orchestration\n"
        "import vpncat.cross_session_runner\n"
        "import vpncat.cross_session_orchestration\n"
        "import vpncat.ablation_orchestration\n"
        "from vpncat.artifacts import _environment_versions\n"
        "from vpncat.cross_session_artifacts import _environment_versions as cross_env\n"
        "assert 'torch' not in sys.modules\n"
        "versions = _environment_versions(include_neural_runtime=False)\n"
        "cross_versions = cross_env(include_neural_runtime=False)\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'torch' in versions\n"
        "assert 'torch' in cross_versions\n"
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="XGBoost optional dependency is not installed",
)
def test_xgboost_fit_survives_after_importing_classical_orchestrator() -> None:
    completed = _run_clean_python(
        "import sys\n"
        "import numpy as np\n"
        "import vpncat.cli\n"
        "import vpncat.orchestration\n"
        "assert 'torch' not in sys.modules\n"
        "from xgboost import XGBClassifier\n"
        "x = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]])\n"
        "y = np.array([0, 0, 0, 0, 1, 1, 1, 1])\n"
        "model = XGBClassifier(n_estimators=2, max_depth=2, n_jobs=1, verbosity=0)\n"
        "model.fit(x, y)\n"
        "assert 'torch' not in sys.modules\n"
    )
    assert completed.returncode == 0, (
        f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )
