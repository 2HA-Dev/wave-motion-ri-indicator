from __future__ import annotations

import os
from typing import Any


def _get_mpi_comm() -> Any | None:
    """
    Return MPI.COMM_WORLD when mpi4py is available and MPI is active.

    The test suite must remain importable in environments where mpi4py is absent.
    """
    try:
        from mpi4py import MPI
    except Exception:
        return None

    if not MPI.Is_initialized():
        return None

    if MPI.Is_finalized():
        return None

    return MPI.COMM_WORLD


def _mpi_rank() -> int:
    comm = _get_mpi_comm()

    if comm is not None:
        return int(comm.rank)

    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MPI_RANK"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)

    return 0


def _mpi_barrier() -> None:
    comm = _get_mpi_comm()

    if comm is not None and comm.size > 1:
        comm.Barrier()


def _trace_enabled() -> bool:
    return os.environ.get("ELASTIC_FWI_TRACE_MPI_TESTS", "0") == "1"


def pytest_runtest_setup(item):
    """
    Synchronise MPI ranks before each test.

    This avoids a rank entering collectives from test k+1 while another rank is
    still executing collectives from test k.
    """
    _mpi_barrier()

    if _trace_enabled():
        print(f"[MPI rank {_mpi_rank()}] START {item.nodeid}", flush=True)


def pytest_runtest_teardown(item, nextitem):
    """
    Synchronise MPI ranks after each test.

    This is intentionally placed in teardown so that tests using collective
    DOLFINx/PETSc operations finish on all ranks before pytest advances.
    """
    if _trace_enabled():
        print(f"[MPI rank {_mpi_rank()}] END   {item.nodeid}", flush=True)

    _mpi_barrier()
