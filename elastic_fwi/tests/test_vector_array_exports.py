from __future__ import annotations

import csv
import json

import numpy as np

from elastic_fwi.inversion import (
    build_vector_array_residual_pair,
    compute_vector_array_residual_series,
)
from elastic_fwi.io import (
    build_vector_array_export_manifest,
    export_vector_array_residual_pair_csv,
    export_vector_array_residual_pair_npz,
    export_vector_array_time_series_npz,
    write_vector_array_manifest,
)
from elastic_fwi.observations import VectorArrayObservationTimeSeries, vector_array_observation
from elastic_fwi.transducers import build_line_transducer_array


def _make_series(values, *, name="gamma2", role="transmission"):
    values = np.asarray(values, dtype=float)

    array = build_line_transducer_array(
        name=name,
        role=role,
        start_x=3.0,
        start_y=9.0 if name == "gamma2" else 1.0,
        end_x=7.0,
        end_y=9.0 if name == "gamma2" else 1.0,
        num_elements=values.shape[1],
        sigma=0.25,
        tag=2 if name == "gamma2" else 1,
    )
    spec = vector_array_observation(array=array)

    return VectorArrayObservationTimeSeries(
        spec=spec,
        steps=np.arange(values.shape[0], dtype=np.int64),
        times=0.1 * np.arange(values.shape[0], dtype=float),
        values=values,
    )


def test_export_vector_array_time_series_npz_is_pickle_free(tmp_path):
    series = _make_series(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    path = export_vector_array_time_series_npz(
        path=tmp_path / "signals.npz",
        series=series,
        label="simulated_transmission",
    )

    with np.load(path, allow_pickle=False) as data:
        assert str(data["kind"]) == "vector_array_observation_time_series"
        assert str(data["label"]) == "simulated_transmission"
        assert str(data["array_name"]) == "gamma2"
        assert data["values"].shape == (2, 2, 2)
        assert data["centers"].shape == (2, 2)
        np.testing.assert_allclose(data["values"], series.values)


def test_export_vector_array_residual_pair_npz_and_csv(tmp_path):
    reference_reflection = _make_series(
        [[[1.0, 2.0], [3.0, 4.0]]],
        name="gamma1",
        role="source_reflection",
    )
    observed_reflection = _make_series(
        [[[0.5, 1.0], [1.5, 2.0]]],
        name="gamma1",
        role="source_reflection",
    )

    reference_transmission = _make_series(
        [[[10.0, 20.0], [30.0, 40.0]]],
        name="gamma2",
        role="transmission",
    )
    observed_transmission = _make_series(
        [[[1.0, 2.0], [3.0, 4.0]]],
        name="gamma2",
        role="transmission",
    )

    reflection = compute_vector_array_residual_series(
        reference=reference_reflection,
        observed=observed_reflection,
    )
    transmission = compute_vector_array_residual_series(
        reference=reference_transmission,
        observed=observed_transmission,
    )

    pair = build_vector_array_residual_pair(
        reflection=reflection,
        transmission=transmission,
    )

    npz_files = export_vector_array_residual_pair_npz(
        output_dir=tmp_path,
        residuals=pair,
    )
    csv_path = export_vector_array_residual_pair_csv(
        path=tmp_path / "vector_array_residuals.csv",
        residuals=pair,
    )

    assert set(npz_files) == {"reflection", "transmission"}

    with np.load(npz_files["reflection"], allow_pickle=False) as data:
        assert str(data["kind"]) == "vector_array_residual_series"
        assert str(data["array_name"]) == "gamma1"
        assert data["values"].shape == (1, 2, 2)

    with np.load(npz_files["transmission"], allow_pickle=False) as data:
        assert str(data["array_name"]) == "gamma2"
        assert data["max_abs_residual"] == 36.0

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert rows[0]["array"] == "gamma1"
    assert rows[0]["component"] == "x"
    assert float(rows[0]["residual"]) == 0.5


def test_vector_array_manifest_is_json_serializable(tmp_path):
    reference = _make_series([[[2.0, 4.0]]])
    observed = _make_series([[[1.0, 1.0]]])

    residual = compute_vector_array_residual_series(
        reference=reference,
        observed=observed,
    )
    pair = build_vector_array_residual_pair(transmission=residual)

    manifest = build_vector_array_export_manifest(
        residuals=pair,
        files={"transmission": "vector_array_transmission_residuals.npz"},
        extra={"receiver_sigma": 0.25},
    )

    path = write_vector_array_manifest(
        path=tmp_path / "vector_array_manifest.json",
        payload=manifest,
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["kind"] == "vector_array_export_manifest"
    assert loaded["observation_kind"] == "vector_array"
    assert loaded["scientific_role"] == "primary_vector_array_observation_chain"
    assert loaded["extra"]["receiver_sigma"] == 0.25
    assert loaded["residuals"]["transmission"]["array_name"] == "gamma2"
