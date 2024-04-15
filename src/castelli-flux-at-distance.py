from star_model import StarModel
import numpy as np
import pandas as pd
import yaml
from utils import setup_logger
from dotenv import load_dotenv
import os

load_dotenv()
DATA = os.environ.get("DATA")


class CastelliFluxAtDistance:
    """
    Class to calculate the flux of stars at varying distances using the Castelli model.

    Parameters:
    ----
    path_to_star_data (str): Path to stellar data such as hip_id, Distance(pc).
    path_to_castelli_model_data (str): Path to data file from star_model.py.
    path_to_save_scattered_model_dir (str): Path to directory where computed data is to be saved.
    wavelength (float): Wavelength in Angstroms used to select the appropriate flux
    column from the Castelli model data. The column name should be formatted as 'Flux{wavelength}',
    e.g., 'Flux1500' for 1500 Angstroms.
    verbose (bool): Verbosity Flag
    """

    def __init__(
        self,
        path_to_star_data,
        path_to_castelli_model_data,
        path_to_save_scattered_model_dir,
        wavelength,
        verbose=False,
    ):
        self.logger = setup_logger(__name__)
        self.path_to_star_data = path_to_star_data
        self.path_to_castelli_model_data = path_to_castelli_model_data
        self.path_to_save_scattered_model_dir = path_to_save_scattered_model_dir
        self.wavelength = wavelength
        self.verbose = verbose

        self.R_sun = 696340  # Radius of the Sun in kilometers
        self.R_sun *= 3.240779289e-14  # Radius of the Sun in parsecs
        self.T_sun = 5778  # Temperature of the Sun in Kelvin
        self.v_mag_sun = 4.83  # Absolute Magnitude of the Sun

    def calculate_castelli_flux_at_distance(self):
        """Calculates and saves the flux at various distances for each star in the dataset."""
        try:
            star_data = pd.read_csv(self.path_to_star_data)
            castelli_model_data = pd.read_csv(self.path_to_castelli_model_data)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
        n_stars = len(star_data)
        distances = np.arange(1, 101, 1)  # pc

        for i in range(n_stars):
            star = star_data.loc[i, "hip_id"]
            if self.verbose:
                self.logger.info(f"Processing Star: {star}")

            flux = castelli_model_data.loc[i, f"Flux{self.wavelength}"]
            star_distance = castelli_model_data.loc[i, "Distance(pc)"]

            flux_angle_model_list = [
                {
                    "Angle": np.degrees(np.arctan(d / star_distance)),
                    "Flux": flux / (d**2),
                }
                for d in distances
            ]

            flux_angle_model_df = pd.DataFrame(flux_angle_model_list)
            flux_angle_model_df.to_csv(
                os.path.join(
                    self.path_to_save_scattered_model_dir,
                    f"flux_angle_model_{star}.csv",
                )
            )


if __name__ == "__main__":
    castelli_flux = CastelliFluxAtDistance(
        path_to_star_data=os.path.join(
            DATA,
            "processed",
            "m8_stellar_data_gaia_hipparcos_with_computed_distance.csv",
        ),
        path_to_model_star_data=os.path.join(DATA, "processed", "flux_data.csv"),
        path_to_save_scattered_model_dir=os.path.join(DATA, "processed"),
        verbose=True,
    )
    castelli_flux.calculate_castelli_flux_at_distance()
