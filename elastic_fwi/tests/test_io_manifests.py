from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from elastic_fwi.io.manifests import (
    read_manifest,
    require_manifest_keys,
    resolve_manifest_path,
    write_manifest,
)


def test_manifest_helpers_write_read_resolve_and_require(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    manifest_path = tmp_path / "nested" / "manifest.json"

    write_manifest(
        manifest_path,
        {
            "kind": "test_manifest",
            "files": {"artifact": "../artifact.txt"},
            "array": np.asarray([1, 2]),
        },
    )

    manifest = read_manifest(manifest_path)
    required = require_manifest_keys(manifest, ["kind", "files.artifact"])
    resolved = resolve_manifest_path(manifest_path, required["files.artifact"], must_exist=True)

    assert manifest["kind"] == "test_manifest"
    assert resolved == artifact

    with pytest.raises(ValueError):
        require_manifest_keys(manifest, ["missing.key"])
