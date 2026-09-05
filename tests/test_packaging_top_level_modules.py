import subprocess
import sys
import tomllib
from pathlib import Path


def test_every_imported_hermes_state_module_is_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    packaged_modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    state_modules = {path.stem for path in repo_root.glob("hermes_state*.py")}
    assert state_modules <= packaged_modules


def test_split_state_registry_imports_from_installed_console_outside_checkout(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import hermes_state_registry"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
