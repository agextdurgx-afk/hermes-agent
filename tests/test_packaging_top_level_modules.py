from pathlib import Path
import tomllib


def test_every_imported_hermes_state_module_is_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    packaged_modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    assert "hermes_state_holders" in packaged_modules
