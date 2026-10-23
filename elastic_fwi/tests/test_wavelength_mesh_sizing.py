from elastic_fwi.meshing import plan_wavelength_mesh_sizing


def test_wavelength_mesh_sizing_computes_resolution_from_lambda_min():
    result = plan_wavelength_mesh_sizing(
        domain_length_x=10.0,
        domain_length_y=10.0,
        source_frequency=1.0,
        rho=4.5,
        C11=16.24,
        C22=18.07,
        C12=6.9,
        C66=4.67,
        target_ppw=10.0,
        near_target_ppw=20.0,
    )

    assert result.lambda_min > 0.0
    assert result.target_h == result.lambda_min / 10.0
    assert result.near_h == result.lambda_min / 20.0
    assert result.nx > 10
    assert result.ny > 10
    assert result.mesh_far_h == result.target_h
    assert result.mesh_near_h == result.near_h


def test_wavelength_mesh_sizing_rejects_invalid_near_ppw():
    try:
        plan_wavelength_mesh_sizing(
            domain_length_x=10.0,
            domain_length_y=10.0,
            source_frequency=1.0,
            rho=4.5,
            C11=16.24,
            C22=18.07,
            C12=6.9,
            C66=4.67,
            target_ppw=20.0,
            near_target_ppw=10.0,
        )
    except ValueError as exc:
        assert "near_target_ppw" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")
