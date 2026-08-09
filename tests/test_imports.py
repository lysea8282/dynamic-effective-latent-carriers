def test_package_imports() -> None:
    import dynamic_effective_carriers
    from dynamic_effective_carriers.carriers import fit_local_edit_delta_pca
    from dynamic_effective_carriers.models import DeterministicBeliefModel
    from dynamic_effective_carriers.simulator import simulate_from

    assert dynamic_effective_carriers.__version__ == "0.1.0"
    assert callable(fit_local_edit_delta_pca)
    assert callable(DeterministicBeliefModel)
    assert callable(simulate_from)
