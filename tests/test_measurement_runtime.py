from pathlib import Path

from measurement.runtime import resolve_project_root


def test_project_root_defaults_to_checkout_containing_measurement_script(tmp_path):
    script = tmp_path / "checkout" / "measurement" / "panel.py"
    assert resolve_project_root(script, {}) == str(script.parent.parent.resolve())


def test_project_root_honours_explicit_runtime_checkout(tmp_path):
    checkout = tmp_path / "deployed"
    script = tmp_path / "source" / "measurement" / "panel.py"
    assert resolve_project_root(script, {"ANIMA_LAB_ROOT": str(checkout)}) == str(checkout)
