from astropy.io import fits
import numpy as np
from scipy import interpolate
import os
import yaml


class ReadCastelli:
    """Reads Castelli-Kurucz model spectra data based on spectral types.

    Args:
    ----
        path_to_ck_data_dir (str): The file path to the directory containing the Castelli-Kurucz FITS files.
        path_to_spectral_type_dict (str): The file path to the spectral type dictionary.
        wave (list[float], optional): The array of wavelengths. Defaults to None.

    Attributes:
    ----
        path_to_spectral_type_dict (str): The file path to the spectral type dictionary.
        temperature_array (np.ndarray): Array of temperatures based on the Model's Temperature Grid.
        wave (list[float] | None): The array of wavelengths.
        spectral_types (dict): Dictionary mapping spectral types to [Effective Temperature, log_g] values.

    Methods:
    ----
        find_closest_temperature(temperature: float) -> int:
            Finds the index of the closest temperature in the temperature array.
        map_log_g(log_g: float) -> str:
            Maps the log g value to a string representation.
        find_ck_model(spectral_type: str) -> Tuple[np.ndarray, np.ndarray]:
            Finds and retrieves the CK model spectra data for the given spectral type.

    Notes:
    ----
        - The Castelli-Kurucz FITS filenames must be in the same format as downloaded.
        - This program only supports the model spectra for [M/H] = 0.0
        - Example: "ckp00_40000.fits"
        - The spectral type dictionary file should be in YAML format and contain mappings of spectral types to
          corresponding [Effective Temperature, log_g] values.
        - The CK model spectra data is loaded from FITS files based on the temperature and log g values of the
          spectral types.
    """

    def __init__(
        self,
        path_to_ck_data_dir: str,
        path_to_spectral_type_dict: str,
        wave: list[float] = None,
    ):
        self.path_to_ck_data_dir = path_to_ck_data_dir
        if not os.path.isdir(self.path_to_ck_data_dir):
            raise ValueError(
                f"The provided path_to_ck_data_dir '{self.path_to_ck_data_dir}' does not exist or is not a directory."
            )

        self.path_to_spectral_type_dict = path_to_spectral_type_dict

        # From Model's Temperature Grid given in Table 1 of https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas/
        temper_1 = np.arange(13000, 3000 - 1, -250)
        temper_2 = np.arange(50000, 13000 - 1, -1000)
        self.temperature_array = np.concatenate((temper_2, temper_1))

        self.wave = wave

        with open(self.path_to_spectral_type_dict, "r") as f:
            self.spectral_types = yaml.safe_load(f)

    def find_closest_temperature(self, temperature: float):
        """Finds the index of the closest temperature in the temperature array.

        Args:
        ----
            temperature (float): The temperature to search for.

        Returns:
        ----
            int: The index of the closest temperature in the temperature array.
        """
        differences = np.abs(self.temperature_array - temperature)
        closest_index = np.argmin(differences)
        return closest_index

    def map_log_g(self, log_g: float):
        """Maps the log g value to a string representation.

        Args:
        ----
            log_g (float): The log g value to map.

        Returns:
        ----
            str: The mapped log g value.
        """
        rounded_log_g = round(log_g * 2) / 2  # Round to the nearest 0.5
        int_log_g = int(rounded_log_g * 10)  # Multiply by 10 and convert to int
        str_log_g = "g" + str(int_log_g)  # Convert to string and add "g" prefix
        return str_log_g

    def find_ck_model(self, spectral_type: str):
        """Finds and retrieves the CK model spectra data for the given spectral type.

        Args:
        ----
            spectral_type (str): The spectral type.

        Returns:
        ----
            Tuple[np.ndarray, np.ndarray]: The CK model spectra data consisting of the wavelength and flux arrays.
        """
        T_eff, log_g = self.spectral_types[spectral_type]
        temp_idx = self.find_closest_temperature(T_eff)
        temperature = self.temperature_array[temp_idx]
        str_log_g = self.map_log_g(log_g)

        filename = os.path.join(self.path_to_ck_data_dir, f"ckp00_{temperature}.fits")

        model_spectra = fits.open(filename)
        image = model_spectra[1]
        ck_wave = image.data["WAVELENGTH"]
        ck_data = image.data[str_log_g]

        if len(self.wave) == 0:
            self.wave = ck_wave
            self.data = ck_data
        else:
            f = interpolate.interp1d(ck_wave, ck_data)
            self.data = f(self.wave)

        model_spectra.close()

        return self.wave, self.data
