import numpy as np
import matplotlib.pyplot as plt
from utils import setup_logger

logger = setup_logger(__name__)


def save_density_to_file(
    save_path: str, distance: np.ndarray, total_number_density_hydrogen: np.ndarray
):
    """
    Saves the calculated hydrogen number density to a specified file.

    Parameters:
    ----
    save_path (str): File path where the density data will be saved.
    distance (np.ndarray): Array of distances corresponding to the density measurements.
    total_number_density_hydrogen (np.ndarray): Array of hydrogen number densities.
    """
    with open(save_path, "w") as f:
        for d, density in zip(distance, total_number_density_hydrogen):
            f.write(f"{d} {density}\n")


def plot_density(distance: np.ndarray, total_number_density_hydrogen: np.ndarray):
    """
    Plots the number density of hydrogen as a function of distance.

    Parameters:
    ----
    distance (ndarray): Array of distances (in parsecs).
    total_number_density_hydrogen (ndarray): Array of hydrogen number densities (in cm^-3).
    """
    plt.plot(distance, total_number_density_hydrogen)
    plt.title("Number Density of Hydrogen (n(HI+HII))")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"n(HI + HII) $(cm^{-3})$")
    plt.show()


def convert_ebv_to_density(
    path_to_dust_density_ebv_data: str,
    save_path: str = None,
):
    """
    Converts E(B-V) values to hydrogen number density using the Bohlin, Savage, and Drake 1978 factor.

    Args:
    ----
    path_to_dust_density_ebv_data (str): Path to file containing distance and E(B-V) data.
    save_path (str, optional): File path to save the converted density data. If not given, data will be plotted.
    """
    pc_in_cm = 3.086e18  # conversion from parsecs to cm

    # Value obtained from paper by Bohlin, Savage and Drake
    total_hydrogen_column_density_ebv_ratio = 5.8e21  # cm^-2

    try:
        # Importing the downloaded dust data
        data = np.loadtxt(path_to_dust_density_ebv_data)
        distance = data[:, 0]  # Distance in pc
        dust_ebv = data[:, 1]  # Dust in units of E(B-V) (mags)

        # Excluding the first element. First element is 0 mags for 0 pc.
        # Will cause a divide by zero error
        if distance[0] == 0:
            distance = distance[1:]
            dust_ebv = dust_ebv[1:]

        distance_in_cm = distance * pc_in_cm  # convert to cm

        dust_ebv_per_cm = dust_ebv / distance_in_cm

        total_number_density_hydrogen = (
            dust_ebv_per_cm * total_hydrogen_column_density_ebv_ratio
        )

        if save_path == None:
            plot_density(distance, total_number_density_hydrogen)
        else:
            save_density_to_file(save_path, distance, total_number_density_hydrogen)
    except Exception as e:
        logger.error(f"An Error Occured: {e}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")

    convert_ebv_to_density(os.path.join(DATA, "green-dust-ebv-2000pc.txt"))
