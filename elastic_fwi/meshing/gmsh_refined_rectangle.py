from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import importlib

from mpi4py import MPI


@dataclass(frozen=True)
class RefinementDisk:
    name: str
    center_x: float
    center_y: float
    radius: float
    target_h: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RefinementDisk name must be non-empty.")

        center_x = float(self.center_x)
        center_y = float(self.center_y)
        radius = float(self.radius)
        target_h = float(self.target_h)

        if not isfinite(center_x) or not isfinite(center_y):
            raise ValueError("RefinementDisk center coordinates must be finite.")

        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("RefinementDisk radius must be strictly positive and finite.")

        if not isfinite(target_h) or target_h <= 0.0:
            raise ValueError("RefinementDisk target_h must be strictly positive and finite.")

        object.__setattr__(self, "center_x", center_x)
        object.__setattr__(self, "center_y", center_y)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "target_h", target_h)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "refinement_disk",
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "target_h": self.target_h,
        }


def _gmshio_module():
    # DOLFINx 0.10 exposes gmsh as dolfinx.io.gmsh.
    # Some older versions used dolfinx.io.gmshio.
    try:
        return importlib.import_module("dolfinx.io.gmshio")
    except Exception:
        return importlib.import_module("dolfinx.io.gmsh")


def _set_threshold_size_options(gmsh, field: int, *, h_min: float, h_max: float) -> None:
    try:
        gmsh.model.mesh.field.setNumber(field, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", h_max)
    except Exception:
        gmsh.model.mesh.field.setNumber(field, "LcMin", h_min)
        gmsh.model.mesh.field.setNumber(field, "LcMax", h_max)


def create_gmsh_refined_rectangle_mesh(
    *,
    comm: MPI.Comm,
    domain_length_x: float,
    domain_length_y: float,
    far_h: float,
    refinement_disks: tuple[RefinementDisk, ...] = (),
    model_rank: int = 0,
):
    Lx = float(domain_length_x)
    Ly = float(domain_length_y)
    far_h = float(far_h)

    if not isfinite(Lx) or Lx <= 0.0:
        raise ValueError("domain_length_x must be strictly positive and finite.")

    if not isfinite(Ly) or Ly <= 0.0:
        raise ValueError("domain_length_y must be strictly positive and finite.")

    if not isfinite(far_h) or far_h <= 0.0:
        raise ValueError("far_h must be strictly positive and finite.")

    gmsh = None
    model = None

    if comm.rank == model_rank:
        import gmsh as _gmsh

        gmsh = _gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.model.add("elastic_fwi_refined_rectangle")
        geo = gmsh.model.geo

        p1 = geo.addPoint(0.0, 0.0, 0.0, far_h)
        p2 = geo.addPoint(Lx, 0.0, 0.0, far_h)
        p3 = geo.addPoint(Lx, Ly, 0.0, far_h)
        p4 = geo.addPoint(0.0, Ly, 0.0, far_h)

        l1 = geo.addLine(p1, p2)
        l2 = geo.addLine(p2, p3)
        l3 = geo.addLine(p3, p4)
        l4 = geo.addLine(p4, p1)

        loop = geo.addCurveLoop([l1, l2, l3, l4])
        surface = geo.addPlaneSurface([loop])

        point_tags: list[int] = []
        for disk in refinement_disks:
            point_tags.append(
                geo.addPoint(
                    disk.center_x,
                    disk.center_y,
                    0.0,
                    disk.target_h,
                )
            )

        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Omega")

        threshold_fields: list[int] = []

        for disk, point_tag in zip(refinement_disks, point_tags, strict=True):
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "PointsList", [point_tag])

            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", disk.radius)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2.5 * disk.radius)
            _set_threshold_size_options(
                gmsh,
                threshold,
                h_min=disk.target_h,
                h_max=far_h,
            )

            threshold_fields.append(threshold)

        if threshold_fields:
            if len(threshold_fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_fields[0])
            else:
                minimum = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", threshold_fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.model.mesh.generate(2)
        model = gmsh.model

    gmshio = _gmshio_module()

    try:
        mesh_data = gmshio.model_to_mesh(model, comm, model_rank, gdim=2)
    finally:
        if comm.rank == model_rank and gmsh is not None:
            gmsh.finalize()

    return mesh_data.mesh
