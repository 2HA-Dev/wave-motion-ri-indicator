from elastic_fwi.inversion.observation_functionals import (
    ObservationFunctionalValues,
    ObservationFunctionalWeights,
    compute_observation_functionals,
    full_aperture_misfit,
    observation_series_misfit,
    reflection_misfit,
    transmission_misfit,
    validate_matching_observation_series,
    weighted_aperture_misfit,
)
from elastic_fwi.inversion.observation_residual_series import (
    ObservationResidualData,
    ObservationResidualSeries,
    compute_observation_residual_data,
    compute_observation_residual_series,
)
from elastic_fwi.inversion.adjoint_observation_sources import (
    AdjointObservationRHSRecord,
    AdjointObservationRHSResult,
    FullAdjointObservationRHSResult,
    assemble_adjoint_observation_rhs_series,
    assemble_adjoint_observation_rhs_vector,
    assemble_full_adjoint_observation_rhs,
)
from elastic_fwi.inversion.adjoint_solver import (
    AdjointEffectiveSolveResult,
    AdjointEffectiveStepRecord,
    solve_adjoint_effective_rhs_series,
)
from elastic_fwi.inversion.newmark_adjoint import (
    NewmarkAdjointStateResult,
    NewmarkAdjointStateStepRecord,
    newmark_middle_operator_action,
    solve_newmark_discrete_adjoint_state,
)
from elastic_fwi.inversion.material_gradient import (
    MaterialDirectionalGradientRecord,
    MaterialDirectionalGradientResult,
    compute_newmark_material_directional_gradient,
    material_effective_operator_action,
    material_middle_operator_action,
    material_newmark_step_forcing,
)
from elastic_fwi.inversion.material_directions import (
    HomogeneousMaterialDirection,
    MaterialPerturbationMatrices,
    assemble_homogeneous_material_direction_matrices,
    assemble_material_perturbation_matrices,
    homogeneous_material_direction_fields,
)

__all__ = [
    "MaterialQualitativeDofImageResult",
    "compute_material_qualitative_dof_image",
    "ObservationFunctionalValues",
    "ObservationFunctionalWeights",
    "compute_observation_functionals",
    "full_aperture_misfit",
    "observation_series_misfit",
    "reflection_misfit",
    "transmission_misfit",
    "validate_matching_observation_series",
    "weighted_aperture_misfit",
    "ObservationResidualData",
    "ObservationResidualSeries",
    "compute_observation_residual_data",
    "compute_observation_residual_series",
    "AdjointObservationRHSRecord",
    "AdjointObservationRHSResult",
    "FullAdjointObservationRHSResult",
    "assemble_adjoint_observation_rhs_series",
    "assemble_adjoint_observation_rhs_vector",
    "assemble_full_adjoint_observation_rhs",
    "AdjointEffectiveSolveResult",
    "AdjointEffectiveStepRecord",
    "solve_adjoint_effective_rhs_series",
    "NewmarkAdjointStateResult",
    "NewmarkAdjointStateStepRecord",
    "newmark_middle_operator_action",
    "solve_newmark_discrete_adjoint_state",
    "MaterialDirectionalGradientRecord",
    "MaterialDirectionalGradientResult",
    "compute_newmark_material_directional_gradient",
    "material_effective_operator_action",
    "material_middle_operator_action",
    "material_newmark_step_forcing",
    "HomogeneousMaterialDirection",
    "MaterialPerturbationMatrices",
    "assemble_homogeneous_material_direction_matrices",
    "assemble_material_perturbation_matrices",
    "homogeneous_material_direction_fields",
]

from .material_imaging import (
    MaterialQualitativeDofImageResult,
    compute_material_qualitative_dof_image,
)

from .material_image_nodes import (
    NodalMaterialImageResult,
    aggregate_dof_image_to_nodes,
)

from .material_image_cells import (
    StructuredCellMaterialImageResult,
    aggregate_nodal_image_to_structured_cells,
)

from .material_image_defect_metrics import (
    CellDefectLocalizationMetrics,
    compute_cell_defect_localization_metrics,
)

from elastic_fwi.inversion.vector_array_residuals import (
    VectorArrayResidualPair,
    VectorArrayResidualSeries,
    build_vector_array_residual_pair,
    compute_vector_array_residual_series,
    reverse_vector_array_residual_series,
)

__all__ += [
    "VectorArrayResidualPair",
    "VectorArrayResidualSeries",
    "build_vector_array_residual_pair",
    "compute_vector_array_residual_series",
    "reverse_vector_array_residual_series",
]

from elastic_fwi.inversion.vector_array_adjoint_sources import (
    VectorArrayAdjointRHSRecord,
    VectorArrayAdjointRHSResult,
    VectorArrayFullAdjointRHSResult,
    assemble_full_vector_array_adjoint_rhs,
    assemble_vector_array_adjoint_rhs_series,
    assemble_vector_array_adjoint_rhs_series_from_operator,
    assemble_vector_array_adjoint_rhs_vector,
    assemble_vector_array_adjoint_rhs_vector_from_operator,
)

__all__ += [
    "VectorArrayAdjointRHSRecord",
    "VectorArrayAdjointRHSResult",
    "VectorArrayFullAdjointRHSResult",
    "assemble_full_vector_array_adjoint_rhs",
    "assemble_vector_array_adjoint_rhs_series",
    "assemble_vector_array_adjoint_rhs_series_from_operator",
    "assemble_vector_array_adjoint_rhs_vector",
    "assemble_vector_array_adjoint_rhs_vector_from_operator",
]

from elastic_fwi.inversion.residual_wave_interaction_image import (
    SingleSourceVectorArrayFullRHS,
    compute_residual_wave_interaction_image_dg0,
    make_single_source_vector_array_full_rhs,
    make_zero_vector_array_rhs_result,
)
