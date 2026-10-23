from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elastic_fwi.materials.defects import CircularDefectSpec, DefectScenario


def _as_float(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise ValueError(f"Missing key {key!r} in defect specification.")

    return float(data[key])


def load_defect_scenario_json(path: str | Path) -> tuple[DefectScenario, dict[str, Any]]:
    """
    Load a multi-disk defect scenario from a JSON file.

    Expected format:
        {
          "name": "scenario_name",
          "background_factor": 1.0,
          "conflict_policy": "last",
          "defects": [
            {
              "name": "defect_A",
              "center_x": 5.0,
              "center_y": 5.0,
              "radius": 1.0,
              "material_factor": 0.8,
              "norm_order": 2
            }
          ]
        }
    """

    json_path = Path(path)

    with json_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("Defect JSON root must be an object.")

    scenario_name = str(payload.get("name", json_path.stem))
    raw_defects = payload.get("defects")

    if not isinstance(raw_defects, list) or len(raw_defects) == 0:
        raise ValueError("Defect JSON must contain a non-empty 'defects' list.")

    defects: list[CircularDefectSpec] = []

    for index, item in enumerate(raw_defects):
        if not isinstance(item, dict):
            raise ValueError(f"Defect entry #{index} must be an object.")

        name = str(item.get("name", f"defect_{index}"))
        center_x = _as_float(item, "center_x")
        center_y = _as_float(item, "center_y")
        radius = _as_float(item, "radius")
        has_common_factor = (
            "material_factor" in item
            or "factor" in item
        )
        has_component_factor = (
            "rho_factor" in item
            or "stiffness_factor" in item
        )

        if has_common_factor and has_component_factor:
            raise ValueError(
                f"Defect entry #{index} mixes material_factor/factor with "
                "rho_factor/stiffness_factor."
            )

        norm_order = int(item.get("norm_order", 2))

        if has_component_factor:
            if "rho_factor" not in item or "stiffness_factor" not in item:
                raise ValueError(
                    f"Defect entry #{index} must provide both rho_factor "
                    "and stiffness_factor."
                )

            defect = CircularDefectSpec(
                name=name,
                center=(center_x, center_y),
                radius=radius,
                norm_order=norm_order,
                rho_factor=float(item["rho_factor"]),
                stiffness_factor=float(item["stiffness_factor"]),
            )
        else:
            material_factor = float(
                item.get("material_factor", item.get("factor", 0.8))
            )
            defect = CircularDefectSpec(
                name=name,
                center=(center_x, center_y),
                radius=radius,
                material_factor=material_factor,
                norm_order=norm_order,
            )

        defects.append(defect)

    scenario = DefectScenario(name=scenario_name, defects=tuple(defects))

    metadata = {
        "json_path": str(json_path),
        "name": scenario.name,
        "num_defects": len(scenario.defects),
        "background_factor": float(payload.get("background_factor", 1.0)),
        "conflict_policy": str(payload.get("conflict_policy", "last")),
        "defects": [
            {
                "name": defect.name,
                "center_x": defect.center_x,
                "center_y": defect.center_y,
                "radius": defect.radius,
                "material_factor": defect.material_factor,
                "rho_factor": defect.rho_factor,
                "stiffness_factor": defect.stiffness_factor,
                "norm_order": defect.norm_order,
            }
            for defect in scenario.defects
        ],
    }

    return scenario, metadata
