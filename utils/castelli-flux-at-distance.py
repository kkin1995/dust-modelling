from star_model import StarModel
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
import os

load_dotenv()
DATA = os.environ.get("DATA")


class CastelliFluxAtDistance:
    def __init__(
        self,
        path_to_star_data,
        path_to_model_star_data,
        path_to_spectral_type_dict,
        verbose=False,
    ):
        self.path_to_star_data = path_to_star_data
        self.path_to_model_star_data = path_to_model_star_data
        self.path_to_spectral_type_dict = path_to_spectral_type_dict
        self.verbose = verbose

        self.R_sun = 696340  # km
        self.R_sun *= 3.240779289e-14  # pc
        self.T_sun = 5778  # K
        self.v_mag_sun = 4.83  # Absolute Magnitude

        with open(path_to_spectral_type_dict, "r") as f:
            self.spectral_type_dict = yaml.load(f, Loader=yaml.FullLoader)

    def calculate_castelli_flux_at_distance(self):
        star_data = pd.read_csv(self.path_to_star_data)
        n_stars = len(star_data)
        distances = np.arange(1, 101, 1)  # pc

        for i in range(n_stars):
            star = star_data.loc[i, "hip_id"]
            if self.verbose:
                print(f"Star: {star}")

            df = pd.read_csv(self.path_to_model_star_data)
            flux = df.loc[i, "Flux1500"]
            star_distance = df.loc[i, "Distance(pc)"]

            # Uncomment the following line if you want to scale the flux by the size of the star
            # v_mag = star_data.loc[i, "hip_V"]
            # spectral_type = star_data.loc[i, "hip_spectral_type"][0:2]
            # star_temperature, log_g = self.spectral_type_dict[spectral_type]
            # radius_squared = (
            #     (self.R_sun**2)
            #     * (2.5 ** (self.v_mag_sun - v_mag))
            #     * ((self.T_sun**4) / (star_temperature**4))
            # )

            # Uncomment the following line if you want to scale the flux by the magnitude of the star
            # scale *= (
            #     3.64e-9 * 10 ** (-0.4 * (v_obs - 3.1 * ebv)) / vflux
            # )

            flux_angle_model_list = []
            for distance in distances:
                flux_copy = np.copy(flux)
                flux_angle_model = {}

                angle = np.arctan(distance / star_distance)

                # Uncomment the following line if you want to scale the flux by the size of the star
                # flux_copy *= (radius_squared) / (distance**2)
                flux_copy *= 1 / (distance**2)

                flux_angle_model["Angle"] = np.degrees(angle)
                flux_angle_model["Flux"] = flux_copy

                flux_angle_model_list.append(flux_angle_model)

            flux_angle_model_df = pd.DataFrame(flux_angle_model_list)
            flux_angle_model_df.to_csv(
                os.path.join(DATA, "processed", f"flux_angle_model_{star}.csv")
            )


if __name__ == "__main__":
    wave = np.linspace(1000, 11000, num=10000)
    path_to_spectral_type_dict = os.path.join(
        DATA, "raw", "spectral_type_temperature.yaml"
    )
    path_to_star_data = os.path.join(
        DATA, "processed", "m8_stellar_data_gaia_hipparcos_with_computed_distance.csv"
    )
    path_to_model_star_data = os.path.join(DATA, "processed", "flux_data.csv")

    castelli_flux = CastelliFluxAtDistance(
        path_to_star_data,
        path_to_model_star_data,
        path_to_spectral_type_dict,
        verbose=True,
    )
    castelli_flux.calculate_castelli_flux_at_distance()
