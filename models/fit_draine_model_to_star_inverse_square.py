import numpy as np
import pandas as pd
import os
import glob
import re


class FitDraineModelToStellarFlux:
    """
    A class to find the best fit Draine Infrared Emission Model for model scattered stellar flux at different angular distances from a star.

    Parameters:
    ----

    path_to_draine_models (str): Path to the directory where the downloaded Draine Infrared Emission Models are stored.
    path_to_stellar_model_flux (str): Path to the file where the output of StarModel.extract_flux_data() is stored.
    path_to_angle_flux_data (str): Path to the directory where the file containing flux as a function of the angular distance is stored. Should not point to file, only to immediate parent directory. Filename should be in format "flux_angle_model_{star_id}.csv".
    scale_factor_radiation_field (float): Scale factor for applying to radiation field in Draine's models. Found in Table A3 of Mathis et. al. (1983).
    verbose (bool): Verbosity Flag

    Usage:
    ----

    # Set up the file paths and parameters
    path_to_draine_models = /path/to/draine_models_directory/
    path_to_stellar_model_flux = /path/to/stellar_model_flux_csv_file
    path_to_angle_flux_data = /path/to/angle_flux_data_directory/
    scale_factor_radiation_field = 3.48e-2 * 1e-4  # ergs cm-2 s-1 A-1

    # Create an instance of FitDraineModelToStellarFlux
    fit_object = FitDraineModelToStellarFlux(
        path_to_draine_models,
        path_to_stellar_model_flux,
        path_to_angle_flux_data,
        wavelength,
        scale_factor_radiation_field,
        verbose=True,
    )

    # Fit the stellar flux model to Draine's model
    df = fit_object.fit_model_stellar_flux_to_draine_model()

    """

    def __init__(
        self,
        path_to_draine_models: str,
        path_to_stellar_model_flux: str,
        path_to_angle_flux_data: str,
        scale_factor_radiation_field: float,
        verbose: bool = False,
    ):
        self.path_to_draine_models = path_to_draine_models
        self.path_to_stellar_model_flux = path_to_stellar_model_flux
        self.path_to_angle_flux_data = path_to_angle_flux_data
        self.scale_factor_radiation_field = scale_factor_radiation_field
        self.verbose = verbose

        self.speed_of_light = 3e8  # m s-1

    def count_lines_in_file(self, file_path: str) -> int:
        """
        Method to count number of lines in a file.

        Parameters:
        ----

        file_path (str): Path to the file whose lines are to be counted.

        Returns:
        ----

        line_count (int): Count of lines in the file.
        """
        with open(file_path, "r") as file:
            line_count = 0
            for _ in file:
                line_count += 1
        return line_count

    def read_avg_rad_field(self, draine_models) -> tuple[np.ndarray, np.ndarray]:
        """
        Method to read the average radiation field from Draine Models and associate it with corresponding models.

        Returns:
        ----

        avg_rad_field_list (np.ndarray): Array of average radiation fields.
        corresponding_draine_models (np.ndarray): Array of file paths for Draine models corresponding to the average radiation fields.
        """
        avg_rad_field_list = np.array([])
        corresponding_draine_models = np.array([])
        for file_path in draine_models:
            number_of_lines = self.count_lines_in_file(file_path)

            if number_of_lines != 1:  # To verify that we are not reading an empty file.
                with open(file_path, "r") as f:
                    for i, line in enumerate(f):
                        if i == 6:
                            avg_rad_field = line
                            break

                avg_rad_field = float(avg_rad_field.split("=")[0].strip())

                avg_rad_field_list = np.append(avg_rad_field_list, avg_rad_field)
                corresponding_draine_models = np.append(
                    corresponding_draine_models, file_path
                )

        return avg_rad_field_list, corresponding_draine_models

    def read_draine_model_filenames(self) -> list:
        """
        Method to read the file names of the Draine models for the Milky Way with R_V = 3.1

        Returns:
        ----

        draine_models_for_mw: List of Draine Model File Names
        """
        draine_models_sub_dir = [x[0] for x in os.walk(self.path_to_draine_models)][1:]
        draine_models_for_mw = []
        for i in range(len(draine_models_sub_dir)):
            for file in glob.glob(os.path.join(draine_models_sub_dir[i], "*.txt")):
                if re.search("MW3\.1", file):
                    draine_models_for_mw.append(file)

        return draine_models_for_mw

    @staticmethod
    def read_star_ids(path_to_stellar_model_flux) -> np.array:
        stellar_model_flux = pd.read_csv(path_to_stellar_model_flux)
        return stellar_model_flux.loc[:, "Star"].to_numpy()

    def _read_flux_angle_data(self, star_id: str) -> pd.DataFrame:
        angle_flux_df = pd.read_csv(
            os.path.join(
                self.path_to_angle_flux_data, f"flux_angle_model_{star_id}.csv"
            )
        )
        return angle_flux_df

    def _compare_flux_with_rad_field(
        self,
        star_id: str,
        angle_flux_df: pd.DataFrame,
        avg_rad_field_list: np.ndarray,
        corresponding_draine_models: np.ndarray,
    ) -> list:
        angles = angle_flux_df.loc[:, "Angle"]
        fluxes = angle_flux_df.loc[:, "Flux"]
        to_save_list = []
        for jdx in range(len(angles)):
            to_save = {}

            angle = angles[jdx]
            flux = fluxes[jdx]

            if self.verbose:
                print(f"-- Angle: {angle}")

            difference_between_flux_and_radiation_field = np.abs(
                flux - avg_rad_field_list
            )
            min_difference = difference_between_flux_and_radiation_field[0]
            corresponding_min_draine_model = corresponding_draine_models[0]
            corresponding_min_avg_radiation_field = avg_rad_field_list[0]

            for ldx, difference in enumerate(
                difference_between_flux_and_radiation_field
            ):
                if difference < min_difference:
                    min_difference = difference
                    corresponding_min_draine_model = corresponding_draine_models[ldx]
                    corresponding_min_avg_radiation_field = avg_rad_field_list[ldx]

            to_save["Star"] = star_id
            to_save["Angle"] = angle
            to_save["Flux"] = flux
            to_save["Avg_Rad"] = corresponding_min_avg_radiation_field
            to_save["Difference"] = min_difference
            to_save["Draine_Model"] = corresponding_min_draine_model

            to_save_list.append(to_save)

        return to_save_list

    def fit_model_stellar_flux_to_draine_model(self) -> pd.DataFrame:
        """
        Method to fit the stellar flux model to Draine's model.

        Returns:
        ----

        pd.DataFrame: Dataframe containing the star id, angular distance from star, EUV flux, average radiation field,
                      difference between the flux and radiation field and the corresponding best fit Draine model.
        """
        star_ids = FitDraineModelToStellarFlux.read_star_ids(
            self.path_to_stellar_model_flux
        )

        draine_models_for_mw = self.read_draine_model_filenames()

        avg_rad_field_list, corresponding_draine_models = self.read_avg_rad_field(
            draine_models_for_mw
        )

        avg_rad_field_list *= self.scale_factor_radiation_field

        for idx in range(len(star_ids)):
            star = star_ids[idx]

            if self.verbose:
                print(f"Star: {star}")

            angle_flux_df = self._read_flux_angle_data(star)

            to_save_list = self._compare_flux_with_rad_field(
                star, angle_flux_df, avg_rad_field_list, corresponding_draine_models
            )

            df = pd.DataFrame(to_save_list)
            if self.verbose:
                print(f"--- Saving Best Fit Draine Model for {star}")
            df.to_csv(
                os.path.join(DATA, "processed", f"best_fit_draine_models_{star}.csv")
            )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")

    path_to_draine_models = os.path.join(DATA, "raw", "draine-ir-emission-models")
    path_to_stellar_model_flux = os.path.join(DATA, "processed", "flux_data.csv")
    path_to_angle_flux_data = os.path.join(DATA, "processed")
    scale_factor_radiation_field = 3.48e-2 * 1e-4  # ergs cm-2 s-1 A-1
    fit_object = FitDraineModelToStellarFlux(
        path_to_draine_models,
        path_to_stellar_model_flux,
        path_to_angle_flux_data,
        scale_factor_radiation_field,
        verbose=True,
    )

    df = fit_object.fit_model_stellar_flux_to_draine_model()
