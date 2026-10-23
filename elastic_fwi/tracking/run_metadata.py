from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "elastic_fwi_tracking_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, argparse.Namespace):
        return {str(k): to_jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_metadata(argv: list[str] | None = None) -> dict[str, Any]:
    argv = list(sys.argv if argv is None else argv)
    command_line = " ".join(argv)
    return {
        "argv": argv,
        "command_line": command_line,
        "command_line_hash": stable_hash(command_line),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
    }


def _run_git_command(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    cwd = Path.cwd() if cwd is None else Path(cwd)

    commit = _run_git_command(["rev-parse", "HEAD"], cwd=cwd)
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    root = _run_git_command(["rev-parse", "--show-toplevel"], cwd=cwd)
    status = _run_git_command(["status", "--porcelain"], cwd=cwd)

    return {
        "available": commit is not None,
        "commit": commit,
        "branch": branch,
        "root": root,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}

    for name in ("numpy", "scipy", "mpi4py", "petsc4py", "dolfinx", "ufl", "basix"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", None)
        except Exception:
            packages[name] = None

    return {
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:5]),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "CONDA_DEFAULT_ENV",
                "VIRTUAL_ENV",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PETSC_ARCH",
                "PETSC_DIR",
            )
            if os.environ.get(key) is not None
        },
    }


def _memory_total_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except Exception:
        return None


def hardware_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "user_name": getpass.getuser(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_total_bytes": _memory_total_bytes(),
    }


def parallel_runtime_metadata(comm: Any | None = None) -> dict[str, Any]:
    rank = 0
    size = 1
    mpi_available = False

    if comm is not None:
        rank = int(getattr(comm, "rank", 0))
        size = int(getattr(comm, "size", 1))
        mpi_available = size > 1
    else:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = int(comm.rank)
            size = int(comm.size)
            mpi_available = True
        except Exception:
            pass

    return {
        "mpi_available": mpi_available,
        "rank": rank,
        "size": size,
        "rank0_only_outputs": True,
    }


def build_tracking_metadata(
    *,
    output_dir: str | Path,
    args: argparse.Namespace | Mapping[str, Any] | None = None,
    comm: Any | None = None,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    elapsed_seconds: float | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)

    command = command_metadata()
    git = git_metadata()
    environment = environment_metadata()
    hardware = hardware_metadata()
    parallel = parallel_runtime_metadata(comm)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "output_dir": str(output_dir),
        "command": command,
        "args": to_jsonable(args) if args is not None else None,
        "git": git,
        "environment": environment,
        "hardware": hardware,
        "parallel_runtime": parallel,
        "files": {
            "command_line_txt": str(output_dir / "raw" / "command_line.txt"),
            "command_line_json": str(output_dir / "raw" / "command_line.json"),
            "environment_json": str(output_dir / "environment.json"),
            "hardware_json": str(output_dir / "hardware.json"),
            "git_info_json": str(output_dir / "git_info.json"),
            "parallel_runtime_json": str(output_dir / "parallel_runtime.json"),
            "run_log": str(output_dir / "logs" / "run.log"),
        },
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_tracking_sidecars(
    *,
    output_dir: str | Path,
    tracking: Mapping[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    command = dict(tracking.get("command", {}))
    command_line = str(command.get("command_line", ""))

    command_txt = raw_dir / "command_line.txt"
    command_json = raw_dir / "command_line.json"
    environment_json = output_dir / "environment.json"
    hardware_json = output_dir / "hardware.json"
    git_info_json = output_dir / "git_info.json"
    parallel_json = output_dir / "parallel_runtime.json"
    run_log = logs_dir / "run.log"

    command_txt.write_text(command_line + "\n", encoding="utf-8")
    write_json(command_json, command)
    write_json(environment_json, dict(tracking.get("environment", {})))
    write_json(hardware_json, dict(tracking.get("hardware", {})))
    write_json(git_info_json, dict(tracking.get("git", {})))
    write_json(parallel_json, dict(tracking.get("parallel_runtime", {})))

    if not run_log.exists():
        run_log.write_text(
            "[elastic_fwi tracking] run metadata sidecars generated.\n",
            encoding="utf-8",
        )

    return {
        "command_line_txt": str(command_txt),
        "command_line_json": str(command_json),
        "environment_json": str(environment_json),
        "hardware_json": str(hardware_json),
        "git_info_json": str(git_info_json),
        "parallel_runtime_json": str(parallel_json),
        "run_log": str(run_log),
    }
