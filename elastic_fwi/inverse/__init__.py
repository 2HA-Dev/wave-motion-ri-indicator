"""
Compatibility namespace for the ``elastic_fwi.inverse`` package.

The active inverse/FWI implementation lives in ``elastic_fwi.inversion``.

This package is intentionally kept as a lightweight compatibility layer because
older scripts and import-tree tests may still import modules under
``elastic_fwi.inverse``.

New code should import from ``elastic_fwi.inversion``.
"""

__all__: list[str] = []
