import numpy as np

from dynamic_effective_carriers.carriers import fit_local_edit_delta_pca, oracle_coefficients, reconstruct_delta


def test_rank_four_carrier_smoke() -> None:
    values = np.random.default_rng(7).normal(size=(80, 192))
    fit = fit_local_edit_delta_pca(values, max_rank=8)
    coefficients = oracle_coefficients(values[:3], fit.basis, 4)
    reconstructed = reconstruct_delta(coefficients, fit.basis, 4)
    assert fit.basis.shape == (192, 8)
    assert coefficients.shape == (3, 4)
    assert reconstructed.shape == (3, 192)
    assert np.allclose(fit.basis.T @ fit.basis, np.eye(8), atol=1e-5)
