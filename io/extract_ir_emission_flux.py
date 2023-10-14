import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yaml
from models.fit_draine_model_to_star_inverse_square import FitDraineModelToStellarFlux


class ExtractIREmissionFluxFromDraineModelFiles:
    """
    A class to extract IR Emission Flux from Draine Model Files.

    This class processes files downloaded from 'https://www.astro.princeton.edu/~draine/dust/irem.html'
    Please refer to the above site for file structure and data details.

    Parameters:
    ----
    path_to_stellar_model_flux (str): Path to the file where the output of StarModel.extract_flux_data() is stored.
    path_to_best_fit_draine_models_dir (str): Path to the directory where the best fit Draine models are stored.
    col_density (float): Column Density of dust along the line of sight of observation. Required only if conversion from Jy sr-1 cm2 H-1 to Jy sr-1 is necessary. Must be in units of atoms cm-1. Optional.
    verbose (bool): Verbosity Flag

    Usage:
    ----

    ### Set up the file paths and parameters
    path_to_stellar_model_flux = /path/to/stellar_model_flux_csv_file
    path_to_best_fit_draine_models_dir = /path/to/best_fit_draine_models_directory/

    ### Create an instance of ExtractIREmissionFluxFromDraineModelFiles
    flux_object = ExtractIREmissionFluxFromDraineModelFiles(
        path_to_stellar_model_flux,
        path_to_best_fit_draine_models_dir,
        verbose=True,
    )

    ### Extract the flux
    flux_object.extract_flux()

    """

    JY_TO_MJY = 1e-6  # Conversion Factor to convert from Jansky's to Mega-Jansky's

    def __init__(
        self,
        path_to_stellar_model_flux: str,
        path_to_best_fit_draine_models_dir: str,
        col_density: float = None,
        verbose: bool = False,
    ):
        self.path_to_stellar_model_flux = path_to_stellar_model_flux
        self.path_to_best_fit_draine_models_dir = path_to_best_fit_draine_models_dir
        self.col_density = col_density
        self.verbose = verbose

    def extract_flux(self):
        """
        Method to extract the IR Emission Flux from the Draine Model Files.
        It modifies the best fit Draine model files by adding a new column "IR100" with the IR flux at 100 microns.

        This method specifically processes Draine Model files downloaded from
        'https://www.astro.princeton.edu/~draine/dust/irem.html'.
        It expects the data to start after 61 rows and columns to be delimited by double space ("  ").
        """
        star_ids = FitDraineModelToStellarFlux.read_star_ids(
            self.path_to_stellar_model_flux
        )

        for star in star_ids:
            if self.verbose:
                print(f"Star: {star}")
            filename = os.path.join(
                self.path_to_best_fit_draine_models_dir,
                f"best_fit_draine_models_{star}.csv",
            )
            try:
                df = pd.read_csv(filename, index_col=0)
            except Exception as e:
                print(f"Error: {e}")
                print(
                    f"File Not Found: {filename} or Incorrect Directory: {self.path_to_best_fit_draine_models_dir}"
                )
            n_angles = len(df)
            for i in range(n_angles):
                draine_model_filename = df.loc[i, "Draine_Model"]
                draine_model_df = pd.read_csv(
                    draine_model_filename,
                    skiprows=61,
                    header=None,
                    delimiter="  ",
                    engine="python",
                )
                draine_model_df.columns = ["Wavelength", "nu*dP/dnu", "j_nu"]
                draine_model_df_at_100_microns = draine_model_df.loc[
                    draine_model_df.loc[:, "Wavelength"] == 100.0, :
                ]
                ir_flux_at_100_microns = draine_model_df_at_100_microns.loc[
                    :, "j_nu"
                ].values[0]
                if self.col_density is not None:
                    ir_flux_at_100_microns *= self.col_density  # Convert to Jy sr-1
                ir_flux_at_100_microns *= self.JY_TO_MJY  # Convert to MJy sr-1

                df.loc[i, "IR100"] = ir_flux_at_100_microns

            df.to_csv(filename)


if __name__ == "__main__":
    load_dotenv()
    DATA = os.environ.get("DATA")

    with open(os.path.join(DATA, "processed", "m8_col_density.yaml"), "r") as f:
        m8_col_density_dict = yaml.load(f, Loader=yaml.FullLoader)

    m8_col_density = m8_col_density_dict["N(HI + H2)"]

    path_to_stellar_model_flux = os.path.join(DATA, "processed", "flux_data.csv")
    path_to_best_fit_draine_models_dir = os.path.join(DATA, "processed")

    extract_object = ExtractIREmissionFluxFromDraineModelFiles(
        path_to_stellar_model_flux,
        path_to_best_fit_draine_models_dir,
        m8_col_density,
        verbose=True,
    )
    extract_object.extract_flux()
