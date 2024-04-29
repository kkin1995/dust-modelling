import numpy as np
import matplotlib.pyplot as plt
from utils import setup_logger

logger = setup_logger(__name__)


def save_density_to_file(
    path_to_dust_density_ebv_data: str, total_number_density_hydrogen: np.ndarray
):
    """
    Saves the calculated hydrogen number density to a specified file.

    Parameters:
    ----
    save_path (str): File path where the density data will be saved.
    total_number_density_hydrogen (np.ndarray): Array of hydrogen number densities.
    """
    with open(path_to_dust_density_ebv_data, "r") as file:
        headers = file.readline().strip()  # Read the first line as header
        data = np.loadtxt(file)  # Read the remaining data

    if len(data) != len(total_number_density_hydrogen):
        raise ValueError(
            "Mismatch between the number of distances and the number of density values provided."
        )

    header = headers + " n(HI+HII)(cm^-3)"
    data = np.column_stack((data, total_number_density_hydrogen))
    np.savetxt(
        path_to_dust_density_ebv_data,
        data,
        fmt="%0.4f %0.4f %0.4f",
        header=header,
        comments="",
    )


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


def convert_ebv_to_density(path_to_dust_density_ebv_data: str, plot: bool = False):
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
        data = np.loadtxt(path_to_dust_density_ebv_data, skiprows=2)
        distance = data[:, 0]  # Distance in pc
        dust_ebv = data[:, 1]  # Dust in units of E(B-V) (mags)

        distance_in_cm = distance * pc_in_cm  # convert to cm

        dust_ebv_per_cm = dust_ebv / distance_in_cm

        total_number_density_hydrogen = (
            dust_ebv_per_cm * total_hydrogen_column_density_ebv_ratio
        )

        total_number_density_hydrogen = np.insert(total_number_density_hydrogen, 0, 0)
        distance = np.insert(distance, 0, 0)

        if plot:
            plot_density(distance, total_number_density_hydrogen)
        else:
            save_density_to_file(
                path_to_dust_density_ebv_data, total_number_density_hydrogen
            )
    except Exception as e:
        logger.error(f"An Error Occured: {e}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()
    DATA = os.environ.get("DATA")
    dust_data_path = os.path.join(DATA, "dust_data_green_2019")
    m8_star_data = pd.read_csv(
        os.path.join(DATA, "m8_hipparcos_data_with_distance.csv")
    )
    for idx, row in m8_star_data.iterrows():
        star_id = row["hip_id"]
        logger.info(f"Converting E(B-V) to n(HI) for {star_id}")
        star_id = row["hip_id"]
        dust_data_file = os.path.join(dust_data_path, f"{star_id}.csv")

        convert_ebv_to_density(dust_data_file)
