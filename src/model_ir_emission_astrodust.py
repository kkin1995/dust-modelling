from astropy.io import fits
import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
from scipy.optimize import minimize
import yaml
import os
from src.utils import setup_logger


class DataLoader:
    def __init__(self, config):
        """
        Initializes the DataLoader with configuration for file paths.

        Parameters:
        ----
        config (dict): A dictionary containing file paths and other necessary configurations.
        """
        self.config = config
        self.logger = setup_logger(__name__)

    def load_column_density(self) -> float:
        """
        Loads column density data from a YAML file.

        Returns:
        ----
        column_density (float): The column density value.
        """
        try:
            with open(self.config["path_to_col_density_file"]) as f:
                col_density_data = yaml.safe_load(f)
            column_density = col_density_data["N(HI + H2)"]
            return column_density
        except Exception as e:
            self.logger.error(
                FileNotFoundError(f"Failed to load column density data: {e}")
            )

    def load_astrodust_model(self) -> tuple[float, interpolate.RectBivariateSpline]:
        """
        Loads astrodust model data from a FITS file and creates a spline interpolation model.

        Returns:
        ----
        ir_wave_near_100_microns, emission_spline (tuple): Contains the wavelength in microns near 100 microns, and the spline model.
        """
        try:
            hdul = fits.open(self.config["path_to_astrodust_model"])
            ir_wavelength = hdul[6].data
            ir_wave_near_100_microns = ir_wavelength[
                np.argmin(np.abs(ir_wavelength - 100))
            ]

            emission = hdul[7].data
            log10_u = hdul[5].data
            hdul.close()

            # Using instructions from Hensley and Draine 2022 (https://github.com/brandonshensley/Astrodust/blob/main/notebooks/model_file_tutorial.ipynb)
            emission_spline = interpolate.RectBivariateSpline(
                log10_u, ir_wavelength, emission[:, :, 2]
            )
            return ir_wave_near_100_microns, emission_spline
        except Exception as e:
            self.logger.error(
                FileNotFoundError(f"Failed to load astrodust model data: {e}")
            )

    def load_star_flux_and_distance(self, star_id: int) -> tuple[float, float]:
        """
        Loads stellar flux and distance for a given star from a CSV file.

        Parameters:
        ----
        star_id : int
            The identifier of the star.

        Returns:
        ----
        dstar, sflux (tuple): The distance to the star and the flux at 1100 A.
        """
        try:
            flux_data = pd.read_csv(self.config["path_to_stellar_model_flux"])

            dstar = flux_data.loc[
                flux_data.loc[:, "Star"] == star_id, "Distance(pc)"
            ].values[0]
            sflux = (
                flux_data.filter(regex=("Flux.*"))
                .loc[flux_data.loc[:, "Star"] == star_id, "Flux1100"]
                .values[0]
            )
            return dstar, sflux
        except FileNotFoundError as e:
            self.logger.error(
                FileNotFoundError(f"Failed to load star flux and distance data: {e}")
            )
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Data loading issue: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

    def load_observed_ir_data(self, star_id: int) -> pd.DataFrame:
        """
        Loads observed IR data for a given star from a CSV file.

        Parameters:
        ----
        star_id : int
            The identifier of the star.

        Returns:
        ----
        ir_obs_binned_data (pd.DataFrame): A DataFrame containing observed IR data.
        """
        try:
            path = os.path.join(
                self.config["path_to_binned_ir_data_dir"],
                f"{star_id}_binned_ir.csv",
            )
            ir_obs_binned_data = pd.read_csv(path)
            return ir_obs_binned_data
        except Exception as e:
            self.logger.error(
                FileNotFoundError(f"Failed to load observed IR data: {e}")
            )

    def load_dust_density_data(self) -> np.ndarray:
        """
        Loads dust density data from a text file.

        Returns:
        ----
        dust_density (np.ndarray): An array containing distance and dust density data.
        """
        try:
            dust_density = np.loadtxt(self.config["path_to_dust_density_file"])
            return dust_density
        except Exception as e:
            self.logger.error(
                FileNotFoundError(f"Failed to load dust density data: {e}")
            )


class IREmissionModeler:
    """
    Models and optimizes infrared (IR) emission from interstellar dust around a specific star, using astrophysical data.

    This class handles the loading of necessary astronomical data, performs simulations of light scattering off dust particles, and optimizes parameters to fit the observed IR emissions data.

    Attributes:
        star_id (int): The HIPPARCOS identifier for the star being modeled.
        params (list[float]): Initial guess for the optimization parameters [albedo, phase factor].
        config (dict): Configuration containing paths to data files.

    Constants:
        SPEED_OF_LIGHT (float): Speed of light in meters per second, used in calculations involving light travel.
        SPEED_OF_LIGHT_ANGSTROMS (float): Speed of light converted to Angstroms per second.
        SCALE_FACTOR_RADIATION_FIELD (float): Scaling factor for radiation field intensity based on Mathis et al., 1983.

    Usage:
        config = {'path_to_data': 'path/to/datafiles'}
        modeler = IREmissionModeler(star_id=12345, initial_params=[0.5, 0.1], config=config)
        modeler.optimize(optimizer='Nelder-Mead')
    """

    SPEED_OF_LIGHT = 3e8  # m/s
    SPEED_OF_LIGHT_ANGSTROMS = SPEED_OF_LIGHT * 1e10
    SCALE_FACTOR_RADIATION_FIELD = (
        3.48e-2 * 1e-4
    )  # ergs cm-2 s-1 A-1 # Mathis et al 1983

    def __init__(self, star_id: int, initial_params: list[float], config: dict):
        self.star_id = star_id
        self.params = initial_params
        self.config = config

        self.logger = setup_logger(__name__)
        self.data_loader = DataLoader(self.config)
        self.load_data()

    def load_data(self):
        """
        Loads all necessary observational and model data required for IR emission calculations from configured sources.

        This method sets up the model with appropriate astronomical data, including stellar fluxes, dust densities, and other relevant properties from various files specified in the `config` dictionary.

        Raises:
            RuntimeError: If any data files are missing or corrupt, or if data loading otherwise fails.
        """
        try:
            # Loading the column density in the M8 sightline from file
            self.column_density = self.data_loader.load_column_density()

            # Loading arrays from the Astrodust Model
            self.ir_wave_near_100_microns, self.emission_spline = (
                self.data_loader.load_astrodust_model()
            )

            # Loading HIP ID values of Stars in M8
            self.dstar, self.sflux = self.data_loader.load_star_flux_and_distance(
                self.star_id
            )

            # Loading Observed IR Data
            self.ir_obs_binned_data = self.data_loader.load_observed_ir_data(
                self.star_id
            )
            self.ir_obs_angles = self.ir_obs_binned_data.loc[:, "Angle"]
            self.observed_ir100 = self.ir_obs_binned_data.loc[:, "IR100"]

            # Loading Dust Density Data
            self.dust_density = self.data_loader.load_dust_density_data()
            self.dust = self.dust_density.transpose()[1]
        except Exception as e:
            self.logger.error(RuntimeError(f"Data loading error: {e}"))

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
    ) -> np.ndarray:
        """
        Simulates the single scattering of light from a star off interstellar dust particles and calculates the resulting flux.

        Parameters:
        ----
        dust (np.ndarray): Dust density values along the line of sight, in atoms cm-3.
        dsun (np.ndarray): Distances from the sun for each dust density value, in parsecs.
        dstar (float): Distance to the star, in parsecs.
        sflux (float): Stellar flux value, in ergs cm-2 s-1 A-1.
        sigma (float): Cross-section value at the given wavelength, in cm2/H.
        albedo (float): Albedo or scattering coefficient of the particles.
        phase (float): Phase function describing the directional distribution of scattered light.
        angle (float): Angle between star's coordinates and observation point's coordinates.

        Returns:
        flux (np.ndarray): The calculated flux values resulting from the scattering event, in the same units as the input flux.

        Raises:
            ValueError: If any inputs are out of expected range.
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
        self.ir_wave_near_100_microns_angstroms = self.ir_wave_near_100_microns * 1e4
        irem = self.emission_spline.ev(
            np.log10(flux / IREmissionModeler.SCALE_FACTOR_RADIATION_FIELD),
            self.ir_wave_near_100_microns,
        )  # ergs s-1 sr-1 H-1
        irem = (irem * self.column_density) / (
            self.ir_wave_near_100_microns_angstroms
        )  # I_\lambda # ergs cm-2 s-1 A-1 sr-1
        irem *= (
            (self.ir_wave_near_100_microns_angstroms) ** 2
        ) / IREmissionModeler.SPEED_OF_LIGHT_ANGSTROMS  # I_\nu # ergs cm-2 s-1 Hz-1 sr -1
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
        if len(params) != 2:
            self.logger.error(
                ValueError(
                    "Expected two parameters for fitting: albedo and phase factor."
                )
            )

        chisq = 0.0
        a, g = params

        dsun = np.linspace(0, 2000, 99)

        sigma = 1.840  # Extinction Cross Section at 1100 A from Draine
        sigma *= 1e-21

        try:
            self.model_irem = []
            for _, angle in enumerate(self.ir_obs_angles):
                f = np.sum(
                    self.single_scatter(
                        self.dust, dsun, dstar, sflux, sigma, a, g, angle
                    )
                )
                if f > 0.0:
                    irem = self._calculate_model_irem(f)
                    self.model_irem.append(irem)

            if not self.model_irem:
                return 1e10

            chisq = np.sum(
                ((self.observed_ir100 - self.model_irem) ** 2) / self.model_irem
            )
            chisq /= len(self.observed_ir100)
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")

        return chisq

    def optimize(
        self, optimizer: str, options: dict = {"disp": True, "maxiter": 100}
    ) -> scipy.optimize.OptimizeResult:
        """
        Runs the optimization process to find the best fit parameters for the model using the specified optimization algorithm.

        Parameters:
        ----
        optimizer (str): Name of the optimization algorithm to use.
        options (dict, optional): Dictionary of options specific to the optimizer.

        Updates:
        ----
        self.result (scipy.optimize.OptimizeResult): Stores the result of the optimization process, typically including the best-fit parameters and the value of the objective function.

        Raises:
        ----
        ValueError: If the optimizer is not supported or if the optimization fails.
        """
        self.result = minimize(
            self.fit,
            self.params,
            args=(self.sflux, self.dstar),
            method=optimizer,
            callback=self.callback,
            bounds=[[0, 1], [0, 0.999]],
            options=options,
        )

    def plot_results(
        self,
        params,
        figsize=(10, 6),
        colors=("blue", "orange"),
        markerstyles=("o", "s"),
        grid=True,
        title=None,
        xlabel="Angle (Deg)",
        ylabel="IR Emission (MJy sr-1)",
        legend_loc="best",
        legend_title=None,
        save_filename=None,
    ):
        """
        Generates and displays a plot comparing the modeled IR emission to observed data.

        Parameters:
        ----
        figsize (tuple): Figure dimension (width, height) in inches.
        colors (tuple): Colors for model and observed data points.
        markerstyles (tuple): Marker styles for model and observed data points.
        grid (bool): Whether to display a grid.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend_loc (str): Location of the legend.
        legend_title (str): Title for the legend.
        save_filename (str): If provided, save the plot to this file.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.scatter(
            self.ir_obs_angles,
            self.model_irem,
            s=5,
            c=colors[0],
            marker=markerstyles[0],
            label="Model IR",
        )
        plt.scatter(
            self.ir_obs_angles,
            self.observed_ir100,
            s=5,
            c=colors[1],
            marker=markerstyles[1],
            label="Observed IR",
        )

        if title:
            plt.title(title)
        else:
            plt.title(
                f"IR Emission Modeling - HIP {self.star_id} | a = {params[0]:.2f} | g = {params[1]:.2f}"
            )

        if grid:
            plt.grid(True)

        if legend_title:
            plt.legend(title=legend_title, loc=legend_loc)
        else:
            plt.legend(loc=legend_loc)

        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel("Angles (Deg)")

        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("IR Emission (MJy sr-1)")

        if save_filename:
            plt.savefig(save_filename)
        else:
            plt.show()

    def callback(self, intermediate_result):
        """
        Optional callback function for the optimizer to provide intermediate output.

        Parameters:
        ----
        intermediate_result: The current result of the optimization process.
        """

        chisq = self.fit(intermediate_result.x, self.sflux, self.dstar)
        print(
            f"a = {intermediate_result.x[0]} | g = {intermediate_result.x[1]} | Chi Square = {chisq}"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")

    stars = [88469, 88496, 88506, 88380, 88581, 88560, 88256, 88142, 88705, 88463]
    # stars = [88469]

    params = [0.3, 0.6]

    config = {
        "path_to_stellar_model_flux": os.path.join(DATA, "flux_data_m8.csv"),
        "path_to_astrodust_model": os.path.join(DATA, "green-dust-density-2000pc.txt"),
        "path_to_dust_density_file": os.path.join(
            DATA, "green-dust-density-2000pc.txt"
        ),
        "path_to_ir_data_dir": os.path.join(
            DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/with_angle"
        ),
        "path_to_binned_ir_data_dir": os.path.join(
            DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/binned_ir_data"
        ),
        "path_to_col_density_file": os.path.join(DATA, "m8_col_density.yaml"),
    }

    options = {"disp": True, "maxiter": 1000}

    ir_emission_model_results = []

    for star in stars:
        data = {}
        print(f"Star: {star}")
        data["Star"] = star
        modeler = IREmissionModeler(star, params, config)

        modeler.optimize("Nelder-Mead", options=options)
        result = modeler.result
        data["CHISQ"] = result.fun
        data["a"] = result.x[0]
        data["g"] = result.x[1]
        # modeler.plot_results(
        #     result.x,
        #     save_filename=os.path.join(
        #         DATA,
        #         "model_observed_ir_plots",
        #         f"{star}_model_observed_ir_emission.png",
        #     ),
        # )
        ir_emission_model_results.append(data)

        # print(f"Optimized a = {result.x[0]}")
        # print(f"Optimized g = {result.x[1]}")

    df = pd.DataFrame(ir_emission_model_results)
    df.to_csv(os.path.join(DATA, "ir_emission_model_results.csv"))
