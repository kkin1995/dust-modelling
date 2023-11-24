from astropy.io import fits
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
import yaml
import os


class IREmissionModeler:
    """
    A class to model and optimize infrared (IR) emission from interstellar dust for a given star.

    Attributes:
    ----
        star_id (int): The HIPPARCOS identifier for the star being modeled.
        params (list[float]): Initial guess for the optimization parameters (albedo and phase factor).
        path_to_stellar_model_flux (str): Path to the CSV file containing stellar model flux data.
        path_to_astrodust_model (str): Path to the FITS file containing astrodust model data.
        path_to_dust_density_file (str): Path to the file containing dust density data.
        path_to_ir_data_dir (str): Path to the directory containing IR data.
        path_to_binned_ir_data_dir (str): Path to the directory containing binned IR data.
        path_to_col_density_file (str): Path to the file containing column density data.

    Methods:
    ----
        load_data: Loads necessary data from files and initializes constants.
        single_scatter: Simulates single scattering of light off dust particles.
        calculate_model_irem: Calculates model IR emission based on provided flux.
        fit: Fits the model to observed data using Chi Square minimization.
        optimize: Runs the optimization process to find the best fit parameters.
        plot_results: Generates a plot comparing the modeled IR emission to the observed data.
        callback: A callback function to provide intermediate outputs during optimization.
    """

    def __init__(
        self,
        star_id: int,
        initial_params: list[float],
        path_to_stellar_model_flux: str,
        path_to_astrodust_model: str,
        path_to_dust_density_file: str,
        path_to_ir_data_dir: str,
        path_to_binned_ir_data_dir: str,
        path_to_col_density_file: str,
    ):
        self.star_id = star_id
        self.params = initial_params

        self.path_to_stellar_model_flux = path_to_stellar_model_flux
        self.path_to_astrodust_model = path_to_astrodust_model
        self.path_to_dust_density_file = path_to_dust_density_file
        self.path_to_ir_data_dir = path_to_ir_data_dir
        self.path_to_binned_ir_data_dir = path_to_binned_ir_data_dir
        self.path_to_col_density_file = path_to_col_density_file

        # Defining Constants
        self.speed_of_light = 3e8
        self.speed_of_light_angstroms = self.speed_of_light * 1e10
        self.scale_factor_radiation_field = (
            3.48e-2 * 1e-4
        )  # ergs cm-2 s-1 A-1 # Mathis et al 1983

        self.load_data()

    def load_data(self):
        """
        Loads the observational and model data required for the IR emission calculations.
        This includes column density, astrodust model data, and stellar flux data.
        """
        # Loading the column density in the M8 sightline from file
        with open(self.path_to_col_density_file) as f:
            col_density_data = yaml.load(f, Loader=yaml.FullLoader)

        self.column_density = col_density_data["N(HI + H2)"]

        # Loading arrays from the Astrodust Model
        hdul = fits.open(path_to_astrodust_model)
        ir_wavelength = hdul[6].data
        self.ir_wave_near_100_microns = ir_wavelength[
            np.argmin(np.abs(ir_wavelength - 100))
        ]
        self.ir_wave_near_100_microns_angstroms = self.ir_wave_near_100_microns * 1e4
        emission = hdul[7].data
        log10_u = hdul[5].data
        hdul.close()

        # Loading HIP ID values of Stars in M8
        flux_data = pd.read_csv(self.path_to_stellar_model_flux)

        self.dstar = flux_data.loc[
            flux_data.loc[:, "Star"] == self.star_id, "Distance(pc)"
        ].values[0]
        self.sflux = (
            flux_data.filter(regex=("Flux.*"))
            .loc[flux_data.loc[:, "Star"] == self.star_id, "Flux1100"]
            .values[0]
        )

        self.ir_obs_binned_data = pd.read_csv(
            os.path.join(
                self.path_to_binned_ir_data_dir, f"{self.star_id}_binned_ir_100.csv"
            )
        )

        self.ir_obs_angles = self.ir_obs_binned_data.loc[:, "Angle"]
        self.observed_ir100 = self.ir_obs_binned_data.loc[:, "IR100"]

        self.dust_density = np.loadtxt(self.path_to_dust_density_file)
        self.dust = self.dust_density.transpose()[1]

        # Using instructions from Hensley and Draine 2022 (https://github.com/brandonshensley/Astrodust/blob/main/notebooks/model_file_tutorial.ipynb)
        self.emission_spline = interpolate.RectBivariateSpline(
            log10_u, ir_wavelength, emission[:, :, 2]
        )

    def single_scatter(
        self,
        dust: list | np.ndarray,
        dsun: np.ndarray,
        dstar: float,
        sflux: float,
        sigma: float,
        albedo: float,
        phase: float,
        angle: float,
    ) -> float:
        """
        Perform a single scattering simulation. This simulates the scattering of light from
        a star off of a dust particle and calculates the flux as a result of this event.

        Parameters:
        ----
        dust : list or np.ndarray
            The dust density values along the line of sight. Units: atoms cm-3
        dsun : np.ndarray
            The distances from the sun for each dust density value. Units: pc
        dstar : float
            The distance to the star. Units: pc
        sflux : float
            The stellar flux value. Units: ergs cm-2 s-1 A-1
        sigma : float
            The cross-section value at the given wavelength. Units: cm2 / H
        albedo : float
            The albedo, or scattering coefficient, of the particles.
        phase : float
            The phase function, which describes the directional distribution of scattered light.
        angle : float
            The scattering angle. Units: Degrees

        Returns:
        ----
        float
            The calculated flux resulting from the scattering events.
        """
        radeg = 57.2958
        delta_dist = dsun[1] - dsun[0]
        min_dist = np.sin(angle / radeg) * dstar
        dist = np.sqrt((dsun - dstar * np.cos(angle / radeg)) ** 2 + min_dist**2)
        dangle = np.arcsin(min_dist / dist)
        minpos = np.argmin(dist)  # minpos - index of min value of dist
        if minpos > len(dangle):
            dangle[minpos + 1 :] = np.pi - np.asin(min_dist / dist[minpos + 1 :])

        # Henyey - Greenstein Phase Function

        sca = (albedo / (4 * np.pi)) * (
            (1 - (phase**2))
            / (1 + (phase**2) - (2 * phase * np.cos(dangle))) ** (3 / 2)
        )

        flux = (sflux / (4.0 * np.pi * dist**2)) * sca

        delta_dist *= 3.08e18
        nflux = len(flux)
        for i in range(nflux):
            flux[i] = flux[i] * np.exp(-sigma * np.sum(dust[0:i]) * delta_dist)

        return flux

    def _calculate_model_irem(self, flux: float) -> float:
        """
        Calculates the model infrared emission for a given stellar flux (radiation field).

        Parameters:
        ----
        flux (float): The given stellar flux (radiation field) for which the IR emission is to be calculated.

        Returns:
        ----
        float: The calculated model infrared emission.
        """
        irem = self.emission_spline.ev(
            np.log10(flux / self.scale_factor_radiation_field),
            self.ir_wave_near_100_microns,
        )  # ergs s-1 sr-1 H-1
        irem = (irem * self.column_density) / (
            self.ir_wave_near_100_microns_angstroms
        )  # I_\lambda # ergs cm-2 s-1 A-1 sr-1
        irem *= (
            (self.ir_wave_near_100_microns_angstroms) ** 2
        ) / self.speed_of_light_angstroms  # I_\nu # ergs cm-2 s-1 Hz-1 sr -1
        irem *= 1e23  # Jy sr-1
        irem *= 1e-6  # MJy sr-1

        return irem

    def fit(self, params: list[float], sflux: float, dstar: float) -> float:
        """
        Fits the model to the observed data using Chi Square minimization.

        Parameters:
        ----
        params (list[float]): The parameters to fit, including albedo and phase factor.
        sflux (float): The stellar flux value.
        dstar (float): The distance to the star.

        Returns:
        ----
        float: The resulting chi-square value from the fit.
        """
        chisq = 0.0
        a, g = params

        dsun = np.linspace(0, 2000, 99)

        sigma = 1.840  # Extinction Cross Section at 1100 A from Draine
        sigma *= 1e-21

        self.model_irem = []
        for _, angle in enumerate(self.ir_obs_angles):
            f = np.sum(
                self.single_scatter(self.dust, dsun, dstar, sflux, sigma, a, g, angle)
            )
            if f > 0.0:
                irem = self._calculate_model_irem(f)
                self.model_irem.append(irem)

        if not self.model_irem:
            return 1e10

        chisq = np.sum(((self.observed_ir100 - self.model_irem) ** 2) / self.model_irem)
        chisq /= len(self.observed_ir100)

        return chisq

    def optimize(self, optimizer):
        """
        Runs the optimization process to determine the best model parameters.

        Parameters:
        ----
        optimizer (str): The optimization algorithm to use.

        Returns:
        ----
        OptimizeResult: The result of the optimization process.
        """
        result = minimize(
            self.fit,
            self.params,
            args=(self.sflux, self.dstar),
            method=optimizer,
            callback=self.callback,
            bounds=[[0, 1], [0, 0.999]],
            options={"disp": True, "maxiter": 100},
        )
        return result

    def plot_results(self):
        """
        Generates and displays a plot comparing the modeled IR emission to observed data.
        """
        import matplotlib.pyplot as plt

        plt.scatter(self.ir_obs_angles, self.model_irem, s=5, label="Model IR")
        plt.scatter(self.ir_obs_angles, self.observed_ir100, s=5, label="Observed IR")
        plt.title(f"IR Emission Modeling - HIP {self.star_id}")
        plt.legend()
        plt.xlabel("Angles (Deg)")
        plt.ylabel("IR Emission (MJy sr-1)")
        plt.show()

    def callback(self, intermediate_result):
        """
        Optional callback function for the optimizer to provide intermediate output.

        Parameters:
        ----
        intermediate_result: The current result of the optimization process.
        """
        chisq = self.fit(intermediate_result, self.sflux, self.dstar)
        print(
            f"a = {intermediate_result[0]} | g = {intermediate_result[1]} | Chi Square = {chisq}"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")

    STAR = 88496
    params = [0.8, 0.4]

    path_to_stellar_model_flux = os.path.join(DATA, "processed", "flux_data.csv")
    path_to_astrodust_model = os.path.join(
        DATA, "raw", "ir_data", "astrodust+PAH_MW_RV3.1.fits"
    )
    path_to_dust_density_file = os.path.join(
        DATA, "processed", "green-dust-density-2000pc.txt"
    )
    path_to_ir_data_dir = os.path.join(DATA, "raw", "ir_data", "extracted_data")
    path_to_binned_ir_data_dir = os.path.join(DATA, "derived")

    path_to_col_density_file = os.path.join(DATA, "processed", "m8_col_density.yaml")

    modeler = IREmissionModeler(
        STAR,
        params,
        path_to_stellar_model_flux,
        path_to_astrodust_model,
        path_to_dust_density_file,
        path_to_ir_data_dir,
        path_to_binned_ir_data_dir,
        path_to_col_density_file,
    )

    result = modeler.optimize("L-BFGS-B")
    modeler.plot_results()

    print(f"Optimized a = {result.x[0]}")
    print(f"Optimized g = {result.x[1]}")
