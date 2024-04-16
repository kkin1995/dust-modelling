import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import OptimizeResult
from src.model_ir_emission_astrodust import DataLoader, IREmissionModeler

# Assuming DataLoader and IREmissionModeler are defined in a module named modeler


def test_load_column_density():
    yaml_data = """
    N(HI + H2): 1.23e22
    """
    with patch("builtins.open", new=mock_open(read_data=yaml_data)), patch(
        "yaml.safe_load", return_value={"N(HI + H2)": 1.23e22}
    ) as mock_yaml:
        loader = DataLoader({"path_to_col_density_file": "fake_path.yaml"})
        assert loader.load_column_density() == pytest.approx(1.23e22)
        mock_yaml.assert_called_once()


def test_file_not_found_logging():
    config = {"path_to_col_density_file": "nonexistent.yaml"}

    # Custom side effect to handle different file paths
    def custom_open(path, *args, **kwargs):
        if path == "nonexistent.yaml":
            raise FileNotFoundError("File not found")
        else:
            # For any other file, return a mock file handle
            return mock_open()()

    with patch("builtins.open", side_effect=custom_open) as mock_file:
        # Mock the logger to not create a file
        with patch("logging.FileHandler", new=mock_open()):
            loader = DataLoader(config)
            result = loader.load_column_density()

            # Result should be None since FileNotFoundError should be logged, not raised
            assert result is None

            # Check that the specific file load attempt (not the logger) tried to open the nonexistent file
            # Check that any call to `open` was with 'nonexistent.yaml'
            found_expected_call = any(
                call[0][0] == "nonexistent.yaml" for call in mock_file.call_args_list
            )
            assert (
                found_expected_call
            ), "File 'nonexistent.yaml' was not opened as expected"


def test_single_scatter():
    modeler = IREmissionModeler(star_id=123, initial_params=[0.5, 0.1], config={})
    modeler.dust = np.array([1, 2, 3])
    modeler.dstar = 10
    modeler.sflux = 1e-3
    sigma = 1e-21
    albedo = 0.5
    phase = 0.5
    angle = 45

    result = modeler.single_scatter(
        modeler.dust,
        np.array([5, 10, 15]),
        modeler.dstar,
        modeler.sflux,
        sigma,
        albedo,
        phase,
        angle,
    )
    assert isinstance(
        result, np.ndarray
    )  # More detailed checks should follow based on the expected scientific output


def test_load_data_integration():
    with patch.object(
        DataLoader, "load_column_density", return_value=1.23e22
    ), patch.object(
        DataLoader,
        "load_astrodust_model",
        return_value=(100, MagicMock(spec=RectBivariateSpline)),
    ), patch.object(
        DataLoader, "load_star_flux_and_distance", return_value=(10, 1e-3)
    ), patch.object(
        DataLoader,
        "load_observed_ir_data",
        return_value=pd.DataFrame({"Angle": [45], "IR100": [0.5]}),
    ), patch.object(
        DataLoader, "load_dust_density_data", return_value=np.array([1, 2, 3])
    ):
        modeler = IREmissionModeler(star_id=123, initial_params=[0.5, 0.1], config={})
        modeler.load_data()
        assert modeler.column_density == pytest.approx(1.23e22)
        assert isinstance(modeler.emission_spline, RectBivariateSpline)
        assert modeler.dstar == 10
        assert modeler.sflux == pytest.approx(1e-3)


def test_load_data_sets_sflux():
    # Setup: Define the star ID and initial parameters for the modeler
    star_id = 12345
    initial_params = [0.1, 0.2]  # Example albedo and phase factor
    config = {"path_to_stellar_model_flux": "dummy/path.csv"}

    # Mock the DataLoader's method to return controlled values
    with patch.object(
        DataLoader, "load_star_flux_and_distance", return_value=(100, 0.003)
    ) as mock_method:
        # Instantiate the modeler, which should invoke `load_data`
        modeler = IREmissionModeler(star_id, initial_params, config)

        # Check if DataLoader.load_star_flux_and_distance is called
        mock_method.assert_called_once()

        # Test: Check if sflux is set correctly
        assert modeler.sflux == pytest.approx(0.003)  # The value returned by the mock

        # You can add additional assertions here for dstar and other related attributes
        assert modeler.dstar == 100


def test_optimization():
    star_id = 12345
    initial_params = [0.1, 0.2]
    config = {"some_key": "some_value"}

    # Create a return value for the optimization process
    optimization_result = OptimizeResult({"x": [0.1, 0.2], "fun": 1.0, "success": True})

    # Patching the minimize method globally if specific path fails
    with patch(
        "scipy.optimize.minimize", return_value=optimization_result
    ) as mock_minimize:
        with patch.object(
            DataLoader, "load_star_flux_and_distance", return_value=(100, 0.003)
        ), patch.object(
            DataLoader, "load_column_density", return_value=1e21
        ), patch.object(
            DataLoader, "load_astrodust_model", return_value=(100, MagicMock())
        ), patch.object(
            DataLoader,
            "load_observed_ir_data",
            return_value=pd.DataFrame({"Angle": [45], "IR100": [0.5]}),
        ), patch.object(
            DataLoader, "load_dust_density_data", return_value=np.array([1, 2, 3])
        ):

            from src.model_ir_emission_astrodust import (
                IREmissionModeler,
            )  # Import inside patch context if global fails

            modeler = IREmissionModeler(star_id, initial_params, config)
            modeler.optimize("Nelder-Mead")

            # Verify the call
            mock_minimize.assert_called_once()
            mock_minimize.assert_called_with(
                modeler.fit,
                initial_params,
                args=(modeler.sflux, modeler.dstar),
                method="Nelder-Mead",
                callback=modeler.callback,
                bounds=[[0, 1], [0, 0.999]],
                options={"disp": True, "maxiter": 100},
            )

            # Check the result
            assert modeler.result.success is True
            assert modeler.result.fun == pytest.approx(1.0)
