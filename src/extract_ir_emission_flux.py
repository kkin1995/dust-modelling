import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yaml
from fit_draine_model_to_star_inverse_square import FitDraineModelToStellarFlux
from utils import setup_logger

logger = setup_logger(__name__)


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
    Initialize with file paths and parameters, then call extract_flux() to process files.

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

    def calculate_flux_at_100_microns(self, draine_model_df: pd.DataFrame):
        """
        Calculates the infrared flux at 100 microns from a Draine model as a DataFrame.

        This function extracts the j_nu value corresponding to a wavelength of 100 microns
        from the provided Draine model DataFrame, adjusts it based on the column density
        if provided, and converts the result from Jansky's to Mega-Jansky's.

        Parameters:
        ----
        draine_model_df (pd.DataFrame): DataFrame containing Draine model data with columns
        for 'Wavelength', 'nu*dP/dnu', and 'j_nu'.

        Returns:
        ----
        ir_flux_at_100_microns (float): The infrared flux at 100 microns in Mega-Jansky per steradian (MJy sr-1).
        """
        ir_flux_at_100_microns = draine_model_df.query("Wavelength == 100.0")[
            "j_nu"
        ].iloc[0]

        if self.col_density is not None:
            ir_flux_at_100_microns *= self.col_density  # Convert to Jy sr-1
        ir_flux_at_100_microns *= self.JY_TO_MJY  # Convert to MJy sr-1

        return ir_flux_at_100_microns

    def process_models(self, df: pd.DataFrame):
        """
        Processes each Draine model file specified in the DataFrame to extract and append
        the IR emission flux at 100 microns to the DataFrame.

        This method iterates over rows of the DataFrame which should contain a column
        'Draine_Model' specifying the path to each Draine model file. It reads each model file,
        extracts the IR flux at 100 microns, and appends this value to the DataFrame in a new
        column 'IR100'. If the file is missing or any other error occurs during processing,
        it logs an error.

        Parameters:
        ----
        df (pd.DataFrame): DataFrame containing the paths to Draine model files in the
                        'Draine_Model' column.
        """
        for i, row in df.iterrows():
            draine_model_filename = row["Draine_Model"]
            if not os.path.exists(draine_model_filename):
                logger.error(
                    f"Draine Model File {draine_model_filename} not found. Skipping..."
                )
                continue

            try:
                draine_model_df = pd.read_csv(
                    draine_model_filename,
                    skiprows=61,
                    header=None,
                    delimiter="  ",
                    engine="python",
                )
                draine_model_df.columns = ["Wavelength", "nu*dP/dnu", "j_nu"]

                ir_flux_at_100_microns = self.calculate_flux_at_100_microns(
                    draine_model_df
                )

                df.at[i, "IR100"] = ir_flux_at_100_microns
            except Exception as e:
                logger.error(
                    f"Failed to process model file {draine_model_filename}: {e}"
                )

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
                logger.info(f"Processing Star: {star}")
            filename = os.path.join(
                self.path_to_best_fit_draine_models_dir,
                f"best_fit_draine_models_{star}.csv",
            )
            try:
                df = pd.read_csv(filename, index_col=0)
                self.process_models(df)
                df.to_csv(filename)
            except Exception as e:
                logger.error(f"Error reading pandas DataFrame: {e}")


if __name__ == "__main__":
    load_dotenv()
    DATA = os.environ.get("DATA")

    with open(os.path.join(DATA, "processed", "m8_col_density.yaml"), "r") as f:
        m8_col_density_dict = yaml.load(f, Loader=yaml.FullLoader)

    m8_col_density = m8_col_density_dict["N(HI + H2)"]

    path_to_stellar_model_flux = os.path.join(DATA, "flux_data_m8.csv")
    path_to_best_fit_draine_models_dir = os.path.join(DATA, "processed")

    extract_object = ExtractIREmissionFluxFromDraineModelFiles(
        path_to_stellar_model_flux,
        path_to_best_fit_draine_models_dir,
        m8_col_density,
        verbose=True,
    )
    extract_object.extract_flux()
