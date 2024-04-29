import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from astropy.io import fits
from scipy import interpolate
import yaml

load_dotenv()
DATA = os.environ.get("DATA")


class FitAstrodustModelToStellarFlux:
    """
    A class used to fit stellar flux as a function of angular distance to the radiation field from the
    Astrodust Models (Hensley and Draine 2022).

    Parameters
    ----------
    path_to_stellar_model_flux : str
        Path to the file containing stellar model flux data. Must have a column "Star" containing Star ID's from a catalog.
    path_to_angle_flux_data_dir : str
        Directory path where angle flux data are stored. Must have columns "Angle" and "Flux".
    path_to_astrodust_model : str
        Path to the astrodust model file. FITS file from the website https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3B6E6S
    path_to_save_results : str
        Path where the results will be saved.
    column_density : float
        Column density value for the model. Units are atoms cm-2
    scale_factor_radiation_field : float
        Scaling factor for the radiation field (U). Found in Table A3 of Mathis et. al. (1983).
    speed_of_light : float
        Speed of light, default is 3e8 (m/s).
    verbose : bool
        If True, the class will print out more information.
    """

    def __init__(
        self,
        path_to_stellar_model_flux: str,
        path_to_angle_flux_data_dir: str,
        path_to_astrodust_model: str,
        path_to_save_results: str,
        column_density: float,
        scale_factor_radiation_field: float,
        speed_of_light: float = 3e8,
        verbose: bool = False,
    ):
        self.path_to_stellar_model_flux = path_to_stellar_model_flux
        self.path_to_angle_flux_data_dir = path_to_angle_flux_data_dir
        self.path_to_astrodust_model = path_to_astrodust_model
        self.path_to_save_results = path_to_save_results
        self.column_density = column_density
        self.scale_factor_radiation_field = scale_factor_radiation_field
        self.speed_of_light = speed_of_light
        self.speed_of_light_angstroms = self.speed_of_light * 1e10
        self.verbose = verbose

        flux_data = pd.read_csv(self.path_to_stellar_model_flux)
        self.star_ids = flux_data.loc[:, "Star"].values

        self.hdul = fits.open(self.path_to_astrodust_model)
        self.wavelength = self.hdul[6].data
        self.wave_near_100_microns = self.wavelength[
            np.argmin(np.abs(self.wavelength - 100))
        ]
        self.wave_near_100_microns_angstroms = self.wave_near_100_microns * 1e4
        self.emission = self.hdul[7].data
        self.log10_u = self.hdul[5].data
        self.hdul.close

    def fit(self):
        """
        Fits the flux as a function of angular distance to the scaled radiation field and reads
        the Infrared Emission at a wavelength closest to 100 microns. Save the infrared emission in
        units of MJy sr-1 along with the angle, flux and radiation field U.
        """
        emission_spline = interpolate.RectBivariateSpline(
            self.log10_u, self.wavelength, self.emission[:, :, 2]
        )

        for star_id in self.star_ids:
            if self.verbose:
                print(f"Star: {star_id}")

            filename = os.path.join(
                self.path_to_angle_flux_data_dir, f"flux_angle_model_{star_id}.csv"
            )
            df = pd.read_csv(filename)
            angle = df.loc[:, "Angle"]
            flux = df.loc[:, "Flux"]

            results = []

            num_fluxes = len(flux)
            for i in range(num_fluxes):
                result = {}

                irem_100_microns = emission_spline.ev(
                    np.log10(flux[i] / self.scale_factor_radiation_field),
                    self.wave_near_100_microns,
                )  # (\lambda * I_\lambda) / N_H
                irem_100_microns = (irem_100_microns * self.column_density) / (
                    self.wave_near_100_microns_angstroms
                )  # I_\lambda
                irem_100_microns *= ((self.wave_near_100_microns_angstroms) ** 2) / (
                    self.speed_of_light_angstroms
                )  # I_\nu
                irem_100_microns *= 1e23  # Jy sr-1
                irem_100_microns *= 1e-6  # MJy sr-1

                result["Angle"] = angle[i]
                result["Flux"] = flux[i]
                result["U"] = flux[i] / self.scale_factor_radiation_field
                result["IR100"] = irem_100_microns

                results.append(result)

            results_df = pd.DataFrame(results)
            results_df.to_csv(
                os.path.join(
                    self.path_to_save_results,
                    f"best_fit_astrodust_models_{star_id}.csv",
                ),
                index=False,
            )


if __name__ == "__main__":
    path_to_astrodust_model = os.path.join(
        DATA, "raw", "ir_data", "astrodust+PAH_MW_RV3.1.fits"
    )
    path_to_stellar_model_flux = os.path.join(DATA, "processed", "flux_data.csv")
    path_to_angle_flux_data_dir = os.path.join(DATA, "processed")
    path_to_save_results = os.path.join(DATA, "processed")

    with open(os.path.join(DATA, "processed", "m8_col_density.yaml")) as f:
        col_density_data = yaml.load(f, Loader=yaml.FullLoader)

    N_H = col_density_data["N(HI + H2)"]

    scale_factor_radiation_field = 3.48e-2 * 1e-4  # ergs cm-2 s-1 A-1

    fit_object = FitAstrodustModelToStellarFlux(
        path_to_stellar_model_flux,
        path_to_angle_flux_data_dir,
        path_to_astrodust_model,
        path_to_save_results,
        N_H,
        scale_factor_radiation_field,
        verbose=True,
    )

    fit_object.fit()
