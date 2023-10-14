from io.read_castelli import ReadCastelli
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA = os.environ.get("DATA")


class StarModel:
    """
    Represents a star model for extracting flux measurements from the Castelli-Kurucz model using star data.

    Args:
    ----
        path_to_spectral_type_dict (str): The file path to the spectral type dictionary.
        wave (list[float], optional): The array of wavelengths. Defaults to None.
    """

    def __init__(
        self,
        path_to_spectral_type_dict: str,
        path_to_ck_data_dir: str,
        wave: list[float] = None,
    ):
        self.path_to_spectral_type_dict = path_to_spectral_type_dict
        self.path_to_ck_data_dir = path_to_ck_data_dir
        self.wave = wave
        self.castelli_reader_object = ReadCastelli(
            self.path_to_ck_data_dir, self.path_to_spectral_type_dict, self.wave
        )
        # To Avoid Duplication
        self.DISTANCE_COLUMN = "Distance(pc)"

    def extract_flux_data(
        self,
        path_to_star_data: str,
        extract_at_wavelength: list | float,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extracts flux from Castelli - Kurucz Model using star data.

        Args:
        ----
            path_to_star_data (str): The file path to the star data.
            path_to_spectral_type_dict (str): The file path to the spectral type dictionary.
            extract_at_wavelength (list | float): The wavelength(s) at which flux measurements are to be extracted.
                Can be a single wavelength (float) or a list of wavelengths (floats).
            wave (list, optional): The array of wavelengths. Defaults to None.
            verbose (bool, optional): If True, prints additional information during processing. Defaults to False.

        Returns:
        ----
            pd.DataFrame: A DataFrame containing the extracted flux measurements for each star.

        Raises:
        ----
            ValueError: If the provided data types are invalid or the required files are missing.

        Notes:
        ----
            - The star data file is expected to be a CSV file with columns: "hip_id", "hip_spectral_type",
            "hip_V", "hip_B-V", and "Distance(pc)".
            - The spectral type dictionary file should be compatible with the `ReadCastelli` class.
            - The spectral type dictionary file should be in YAML format and compatible with the `ReadCastelli` class.
            - The YAML file should follow the structure:
                Spectral_Type: [Effective Temperature, log_g]
                Example:
                    "O4": [43000, 3.92]
                    "O6": [39000, 3.92]
                    "O7": [37000, 3.92]
                    "B5": [15000, 4.04]
            - The output DataFrame will contain columns: "Star", "Flux{wavelength}", "EBV", and "Distance(pc)".
            Flux measurements are recorded at each specified wavelength.
        """
        if os.path.isfile(path_to_star_data) == False:
            raise ValueError(f"Star Data File Does Not Exist: {path_to_star_data}")

        if os.path.isfile(self.path_to_spectral_type_dict) == False:
            raise ValueError(
                f"Spectral Type Dictionary File Does Not Exist: {self.path_to_spectral_type_dict}"
            )

        star_data = pd.read_csv(path_to_star_data)
        star_data = star_data.loc[
            :, ["hip_id", "hip_spectral_type", "hip_V", "hip_B-V", self.DISTANCE_COLUMN]
        ]

        n_stars = len(star_data)

        flux_data = []

        for i in range(n_stars):
            flux = {}

            star = star_data.loc[i, "hip_id"]
            if verbose:
                print(f"Star: {star}")

            spectral_type = star_data.loc[i, "hip_spectral_type"][0:2]

            _, sflux = self.castelli_reader_object.find_ck_model(spectral_type)

            b_index = self.find_wavelength_index(4400.0, wave)
            v_index = self.find_wavelength_index(5500.0, wave)
            bflux = sflux[b_index]
            vflux = sflux[v_index]

            b_mag = -2.5 * np.log10(bflux / 4.06, where=bflux > 0)
            v_mag = -2.5 * np.log10(vflux / 3.64, where=vflux > 0)

            ebv = star_data.loc[i, "hip_B-V"] - (b_mag - v_mag)

            if ebv < 0.0:
                ebv = 0.0

            # Converting Castelli Units to Physical Flux Units (ergs cm-2 s-1 A-1)
            scale = np.multiply(3.336e-19, np.divide(np.power(wave, 2), 4 * np.pi))

            # The following code scales the flux by the magnitude of the star.
            # Currently disabled as it may require further validation or modification.
            # Uncomment and test carefully before enabling.
            # scale *= (
            #     3.64e-9 * 10 ** (-0.4 * (star_data.loc[i, "hip_V"] - 3.1 * ebv)) / vflux
            # )

            sflux *= scale

            wavelength_indices = self.extract_wavelength_indices(
                extract_at_wavelength, wave
            )

            if verbose:
                for idx in wavelength_indices:
                    print(f"--Flux at {round(wave[idx])} A: {sflux[idx]}")

            flux["Star"] = star_data.loc[i, "hip_id"]
            for idx in wavelength_indices:
                flux[f"Flux{round(wave[idx])}"] = sflux[idx]
            flux["EBV"] = ebv
            flux[self.DISTANCE_COLUMN] = star_data.loc[i, self.DISTANCE_COLUMN]

            flux_data.append(flux)

        return pd.DataFrame(flux_data)

    def extract_wavelength_indices(
        self, extract_at_wavelength: list | float, wave: list
    ):
        """Extracts the indices of the specified wavelength(s) in the wave array.

        Args:
        ----
            extract_at_wavelength (list | float): The wavelength(s) at which flux measurements are to be extracted.
            wave (list): The array of wavelengths.

        Returns:
        ----
            list: The indices of the specified wavelength(s) in the wave array.
        """
        if isinstance(extract_at_wavelength, list):
            wavelength_indices = [
                self.find_wavelength_index(w, wave) for w in extract_at_wavelength
            ]
        else:
            wavelength_indices = [
                self.find_wavelength_index(extract_at_wavelength, wave)
            ]
        return wavelength_indices

    def find_wavelength_index(self, wavelength: float, wave: list):
        """Finds the index of the closest wavelength in the wave array.

        Args:
        ----
            wavelength (float): The wavelength to search for.
            wave (list): The array of wavelengths.

        Returns:
        ----
            int: The index of the closest wavelength in the wave array.
        """
        return np.argmin(abs(np.subtract(wave, wavelength)))


if __name__ == "__main__":
    # To Avoid Duplication
    DISTANCE_COLUMN = "Distance(pc)"
    path_ck_data_dir = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/m8-dust-modeling/data/raw/castelli-kurucz-models/ckp00/"
    path_to_spectral_type_dict = os.path.join(
        DATA, "raw/spectral_type_temperature.yaml"
    )
    df = pd.read_csv(
        "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/m8-dust-modeling/data/raw/spica_hipparcos_data.csv"
    )
    df.loc[:, DISTANCE_COLUMN] = 1 / (df.loc[:, "parallax(mas)"] * 1e-3)
    star = df.loc[0, "hip_id"]
    df.to_csv(
        "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/m8-dust-modeling/data/raw/spica_hipparcos_data_with_distance.csv"
    )

    path_to_star_data = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/m8-dust-modeling/data/raw/spica_hipparcos_data_with_distance.csv"
    wave = np.linspace(1000, 11000, num=10000)
    extract_at_wavelength = [1100, 1368, 1500, 2300]

    star_model_object = StarModel(path_to_spectral_type_dict, path_ck_data_dir, wave)
    flux_df = star_model_object.extract_flux_data(
        path_to_star_data,
        extract_at_wavelength,
        verbose=True,
    )
    flux_df.to_csv(os.path.join(DATA, "processed", f"flux_data_{star}.csv"))
