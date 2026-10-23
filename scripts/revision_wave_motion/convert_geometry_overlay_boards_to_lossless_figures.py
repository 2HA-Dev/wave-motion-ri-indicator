#!/usr/bin/env python
"""
Lossless packaging of the normalised sensitivity-map galleries with the
Gamma_1/Gamma_2 array overlay (figures of the accompanying article).

SOURCE
------
The published figures ``NormalizedSensitivityGallery_Line.pdf`` and
``NormalizedSensitivityGallery_Box.pdf`` carry the red (Gamma_1,
emitting/reflection array) and green (Gamma_2, transmission array) dashed
array-position overlay described in their caption. Their source rasters are
the boards ``geometry_boards/{line,box}__four_modalities__five_
properties.png`` (4824x3574 px) of the gallery production run
``normalized_sensitivity_map_gallery_geometry_overlay_20260803/``. A
numerical comparison against the JPEG-decoded copy embedded in the published
PDFs gives a mean channel difference of 0.85/255, with sparse high-error
pixels concentrated at edges: the signature of JPEG compression on this exact
raster, not a different image. The ROI36 boards
(``roi36_extension/geometry_boards_roi36``, 4815x3574 px) lack the overlay
and are not a valid source; their nine-pixel width difference is the canvas
space taken by the overlay legend and lines.

``P10_GALLERY_MANIFEST.csv`` in the same folder traces every one of the 40
individual maps that compose the two boards back to its source VTK file by
SHA256, with ROI, normalisation and rendering parameters recorded (see
``verify_source_manifest`` below). This folder is not covered by the master
checksum file of the reference campaigns; it has its own, separate manifest,
which this script checks directly.

WHAT THIS SCRIPT DOES
----------------------
It does NOT re-render the gallery: re-running matplotlib against the
underlying DG0 fields could produce a rendering different from the published
one. Instead it takes the existing, manifest-verified PNG boards
byte-for-byte and re-packages them losslessly:

  1. Verifies all 40 rows of ``P10_GALLERY_MANIFEST.csv`` for the relevant
     geometry: source VTK SHA256 against the file on disk, and individual
     PNG SHA256 against the file on disk. Aborts on any mismatch.
  2. Confirms the alpha channel of each assembled board PNG is uniformly
     opaque (255 everywhere), so the RGBA -> RGB flatten changes no visible
     pixel.
  3. Writes a single-page PDF with the RGB raster embedded as a
     Flate-compressed (not JPEG) Image XObject, built by hand rather than
     through a plotting library, so no re-rendering step can introduce a
     discrepancy. Verified pixel-identical to the source PNG via
     ``pdfimages -png`` extraction.
  4. Derives the matching EPS from that lossless PDF via
     ``gs -sDEVICE=eps2write -dEPSCrop``. Verified pixel-identical via a
     Flate-forced ``pdfwrite`` round trip.

No PDE is solved. The campaign directories are read only.

USAGE
-----
    python path/to/convert_geometry_overlay_boards_to_lossless_figures.py \
        --output-dir path/to/output_directory \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

OVERLAY_DIR = (
    "outputs/Priorité 10 — export_normalized_sensitivity_map_gallery.py/"
    "final_freeze_20260727/normalized_sensitivity_map_gallery_geometry_overlay_20260803"
)
MANIFEST_CSV_NAME = "P10_GALLERY_MANIFEST.csv"

BOARDS = {
    "line": {
        "source": "geometry_boards/line__four_modalities__five_properties.png",
        "canonical_geometry": "line",
        "pdf_name": "NormalizedSensitivityGallery_Line.pdf",
        "eps_name": "NormalizedSensitivityGallery_Line.eps",
    },
    "box": {
        "source": "geometry_boards/box__four_modalities__five_properties.png",
        "canonical_geometry": "box",
        "pdf_name": "NormalizedSensitivityGallery_Box.pdf",
        "eps_name": "NormalizedSensitivityGallery_Box.eps",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_manifest(overlay_dir: Path, repo_root: Path) -> None:
    manifest_path = overlay_dir / MANIFEST_CSV_NAME
    rows = list(csv.DictReader(manifest_path.open()))
    if len(rows) != 40:
        raise SystemExit(f"Expected 40 manifest rows, found {len(rows)} in {manifest_path}")
    for row in rows:
        source_path = Path(row["source_path"])
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.exists():
            raise SystemExit(f"Source VTK missing: {source_path}")
        actual = sha256_file(source_path)
        if actual != row["source_sha256"]:
            raise SystemExit(
                f"Source SHA256 mismatch for {source_path}: "
                f"manifest={row['source_sha256']} actual={actual}"
            )
        individual_png = overlay_dir / row["individual_png"]
        if not individual_png.exists():
            raise SystemExit(f"Individual PNG missing: {individual_png}")
        actual_png = sha256_file(individual_png)
        if actual_png != row["individual_png_sha256"]:
            raise SystemExit(
                f"Individual PNG SHA256 mismatch for {individual_png}: "
                f"manifest={row['individual_png_sha256']} actual={actual_png}"
            )
    print(f"Verified all {len(rows)} rows of {manifest_path} against files on disk.")


def build_lossless_pdf(png_path: Path, pdf_path: Path) -> None:
    im = Image.open(png_path)
    if im.mode == "RGBA":
        alpha_extrema = im.getchannel("A").getextrema()
        if alpha_extrema != (255, 255):
            raise SystemExit(
                f"{png_path} has non-opaque alpha {alpha_extrema}; "
                "flatten would not be lossless."
            )
    im = im.convert("RGB")
    w, h = im.size
    raw = im.tobytes()
    compressed = zlib.compress(raw, level=9)

    content = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode("ascii")
    content_compressed = zlib.compress(content, level=9)

    pdf_objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
            f"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        4: (
            f"<< /Length {len(content_compressed)} /Filter /FlateDecode >>\nstream\n".encode(
                "ascii"
            )
            + content_compressed
            + b"\nendstream"
        ),
        5: (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n"
        ).encode("ascii")
        + compressed
        + b"\nendstream",
    }

    out = bytearray()
    out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = {}
    for num in (1, 2, 3, 4, 5):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii")
        out += pdf_objs[num]
        out += b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 6\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in (1, 2, 3, 4, 5):
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode(
            "ascii"
        )
    )
    pdf_path.write_bytes(bytes(out))


def build_eps_from_pdf(pdf_path: Path, eps_path: Path) -> None:
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dEPSCrop",
            "-sDEVICE=eps2write",
            f"-sOutputFile={eps_path}",
            str(pdf_path),
        ],
        check=True,
    )


def verify_pixel_identical(png_path: Path, pdf_path: Path, eps_path: Path, workdir: Path) -> None:
    import numpy as np

    reference = np.array(Image.open(png_path).convert("RGB"))

    extracted_prefix = workdir / f"{pdf_path.stem}_extracted"
    subprocess.run(
        ["pdfimages", "-png", str(pdf_path), str(extracted_prefix)], check=True
    )
    pdf_extracted = np.array(
        Image.open(f"{extracted_prefix}-000.png").convert("RGB")
    )
    if not np.array_equal(reference, pdf_extracted):
        raise SystemExit(f"{pdf_path} is NOT pixel-identical to {png_path}")

    roundtrip_pdf = workdir / f"{eps_path.stem}_roundtrip.pdf"
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dEPSCrop",
            "-sDEVICE=pdfwrite",
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/FlateEncode",
            f"-sOutputFile={roundtrip_pdf}",
            str(eps_path),
        ],
        check=True,
    )
    eps_extracted_prefix = workdir / f"{eps_path.stem}_extracted"
    subprocess.run(
        ["pdfimages", "-png", str(roundtrip_pdf), str(eps_extracted_prefix)],
        check=True,
    )
    eps_extracted = np.array(
        Image.open(f"{eps_extracted_prefix}-000.png").convert("RGB")
    )
    if not np.array_equal(reference, eps_extracted):
        raise SystemExit(f"{eps_path} is NOT pixel-identical to {png_path}")

    print(f"{pdf_path.name} and {eps_path.name}: verified pixel-identical to {png_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    overlay_dir = args.repo_root / OVERLAY_DIR
    verify_source_manifest(overlay_dir, args.repo_root)

    manifest_entries = {}
    for key, board in BOARDS.items():
        png_path = overlay_dir / board["source"]
        pdf_path = out / board["pdf_name"]
        eps_path = out / board["eps_name"]

        build_lossless_pdf(png_path, pdf_path)
        build_eps_from_pdf(pdf_path, eps_path)
        with tempfile.TemporaryDirectory() as tmp:
            verify_pixel_identical(png_path, pdf_path, eps_path, Path(tmp))

        manifest_entries[key] = {
            "source_png": str(png_path.relative_to(args.repo_root)),
            "source_png_sha256_newly_established": sha256_file(png_path),
            "pdf_sha256": sha256_file(pdf_path),
            "eps_sha256": sha256_file(eps_path),
        }

    manifest = {
        "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "lossless_geometry_overlay_gallery_v1",
        "purpose": (
            "Re-package the manifest-verified geometry-overlay boards "
            "(with the Gamma_1/Gamma_2 array overlay) as lossless "
            "PDF/EPS pairs for Figures 2-3."
        ),
        "source_manifest": str(
            (overlay_dir / MANIFEST_CSV_NAME).relative_to(args.repo_root)
        ),
        "boards": manifest_entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
