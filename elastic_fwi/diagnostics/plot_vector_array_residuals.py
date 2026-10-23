from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load_residual_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    values = np.asarray(data["values"], dtype=float)

    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError(
            f"Expected residual values with shape (num_steps, num_receivers, 2), got {values.shape}."
        )

    if "times" in data:
        times = np.asarray(data["times"], dtype=float)
    else:
        times = np.arange(values.shape[0], dtype=float)

    if times.shape[0] != values.shape[0]:
        raise ValueError("times and values have inconsistent lengths.")

    return {
        "path": str(path),
        "times": times,
        "values": values,
        "num_steps": int(values.shape[0]),
        "num_receivers": int(values.shape[1]),
        "max_abs": float(np.max(np.abs(values))),
        "l2_norm": float(np.linalg.norm(values.reshape(-1))),
    }


def _component_max_abs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # values shape: (num_steps, num_receivers, 2)
    max_x = np.max(np.abs(values[:, :, 0]), axis=1)
    max_y = np.max(np.abs(values[:, :, 1]), axis=1)
    return max_x, max_y


def plot_vector_array_residuals(
    *,
    reflection_npz: Path,
    transmission_npz: Path,
    output_dir: Path,
    stem: str = "vector_array_residuals",
) -> dict[str, Any]:
    reflection = _load_residual_npz(Path(reflection_npz))
    transmission = _load_residual_npz(Path(transmission_npz))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_ref = reflection["times"]
    t_trans = transmission["times"]

    ref_x, ref_y = _component_max_abs(reflection["values"])
    trans_x, trans_y = _component_max_abs(transmission["values"])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(t_ref, ref_x, label=r"$\Gamma_1$ reflection, $u_x$")
    ax.plot(t_ref, ref_y, label=r"$\Gamma_1$ reflection, $u_y$")
    ax.plot(t_trans, trans_x, label=r"$\Gamma_2$ transmission, $u_x$")
    ax.plot(t_trans, trans_y, label=r"$\Gamma_2$ transmission, $u_y$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Max absolute vector residual over receivers")
    ax.set_title("Vector-array residuals on reflection/transmission arrays")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "kind": "vector_array_residual_plot",
        "status": "ok",
        "reflection": {
            "path": reflection["path"],
            "num_steps": reflection["num_steps"],
            "num_receivers": reflection["num_receivers"],
            "max_abs": reflection["max_abs"],
            "l2_norm": reflection["l2_norm"],
        },
        "transmission": {
            "path": transmission["path"],
            "num_steps": transmission["num_steps"],
            "num_receivers": transmission["num_receivers"],
            "max_abs": transmission["max_abs"],
            "l2_norm": transmission["l2_norm"],
        },
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
        },
    }

    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["outputs"]["summary"] = str(summary_path)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot vector-array residual time series.")
    parser.add_argument("--reflection-npz", type=Path, required=True)
    parser.add_argument("--transmission-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", type=str, default="vector_array_residuals")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    summary = plot_vector_array_residuals(
        reflection_npz=args.reflection_npz,
        transmission_npz=args.transmission_npz,
        output_dir=args.output_dir,
        stem=args.stem,
    )

    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
