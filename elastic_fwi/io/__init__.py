"""Input/output helpers.

FEniCSx-dependent vector-array exporters are optional at import time so that
NumPy-only schema and manifest helpers can be tested in lightweight runtimes.
"""

try:  # pragma: no cover - exercised in FEniCSx environments
    from elastic_fwi.io.vector_array_exports import (
        build_vector_array_export_manifest,
        export_vector_array_residual_pair_csv,
        export_vector_array_residual_pair_npz,
        export_vector_array_residual_series_npz,
        export_vector_array_time_series_npz,
        write_vector_array_manifest,
    )
except Exception:  # pragma: no cover - allows NumPy-only imports without ufl/dolfinx
    build_vector_array_export_manifest = None  # type: ignore[assignment]
    export_vector_array_residual_pair_csv = None  # type: ignore[assignment]
    export_vector_array_residual_pair_npz = None  # type: ignore[assignment]
    export_vector_array_residual_series_npz = None  # type: ignore[assignment]
    export_vector_array_time_series_npz = None  # type: ignore[assignment]
    write_vector_array_manifest = None  # type: ignore[assignment]

from elastic_fwi.io.manifests import (
    read_manifest,
    require_manifest_keys,
    resolve_manifest_path,
    write_manifest,
)
from elastic_fwi.io.npz_schema import (
    load_vector_array_residual_npz,
    validate_vector_array_residual_npz,
    write_vector_array_residual_npz,
)

__all__ = [
    "build_vector_array_export_manifest",
    "export_vector_array_residual_pair_csv",
    "export_vector_array_residual_pair_npz",
    "export_vector_array_residual_series_npz",
    "export_vector_array_time_series_npz",
    "write_vector_array_manifest",
    "read_manifest",
    "require_manifest_keys",
    "resolve_manifest_path",
    "write_manifest",
    "load_vector_array_residual_npz",
    "validate_vector_array_residual_npz",
    "write_vector_array_residual_npz",
]
