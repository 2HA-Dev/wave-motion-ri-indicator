from __future__ import annotations

import numpy as np

from elastic_fwi.io.npz_schema import (
    load_vector_array_residual_npz,
    validate_vector_array_residual_npz,
    write_vector_array_residual_npz,
)


def test_write_validate_load_vector_array_residual_npz(tmp_path):
    times = np.linspace(0.0, 1.0, 5)
    values = np.zeros((5, 3, 2), dtype=float)
    values[..., 0] = 1.0
    receiver_coordinates = np.column_stack([np.arange(3), np.zeros(3)])

    path = write_vector_array_residual_npz(
        tmp_path / "residual.npz",
        times,
        values,
        {"aperture": "reflection"},
        receiver_coordinates=receiver_coordinates,
        label="reflection",
    )

    validation = validate_vector_array_residual_npz(path)
    payload = load_vector_array_residual_npz(path)

    assert validation["num_steps"] == 5
    assert validation["num_receivers"] == 3
    assert validation["has_receiver_coordinates"] is True
    assert payload["metadata"]["aperture"] == "reflection"
    assert payload["values"].shape == (5, 3, 2)
