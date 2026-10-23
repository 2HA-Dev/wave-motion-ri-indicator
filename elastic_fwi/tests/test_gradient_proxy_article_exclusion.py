from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

ACTIVE_EXPORTER = (
    ROOT / "elastic_fwi" / "diagnostics" / "export_inverse_time_series_xdmf.py"
)


# This postprocessor is allowed to name the legacy diagnostic
# field only in order to reject it explicitly from article-facing
# method comparisons. It is not an authorised producer or consumer
# of that diagnostic field.
ACTIVE_EXCLUSION_GUARD = (
    ROOT
    / "elastic_fwi"
    / "postprocessing"
    / "single_defect_method_comparison.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gradient_proxy_is_declared_diagnostic_only() -> None:
    text = _read(ACTIVE_EXPORTER)

    assert '"gradient_proxy": gradient_export' in text
    assert '"field_name": "material_gradient_proxy"' in text
    assert '"scientific_role": "diagnostic_proxy_not_primary_gradient"' in text


def test_revision_wave_motion_scripts_do_not_use_gradient_proxy() -> None:
    scripts_dir = ROOT / "scripts" / "revision_wave_motion"

    if not scripts_dir.exists():
        pytest.skip("No revision_wave_motion scripts directory in this checkout.")

    for path in scripts_dir.glob("*.py"):
        text = _read(path)
        assert "gradient_proxy" not in text, str(path)
        assert "material_gradient_proxy" not in text, str(path)

    for path in scripts_dir.glob("*.sh"):
        text = _read(path)
        assert "gradient_proxy" not in text, str(path)
        assert "material_gradient_proxy" not in text, str(path)


def test_active_elastic_fwi_modules_do_not_use_gradient_proxy_except_exporter() -> None:
    allowed = {
        ACTIVE_EXPORTER.resolve(),
        ACTIVE_EXCLUSION_GUARD.resolve(),
    }

    for path in (ROOT / "elastic_fwi").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if "tests" in path.parts:
            continue
        if path.resolve() in allowed:
            continue

        text = _read(path)
        assert "gradient_proxy" not in text, str(path)
        assert "material_gradient_proxy" not in text, str(path)
