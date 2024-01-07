import numpy as np
import matplotlib.pyplot as plt


def save_density_to_file(save_path, distance, total_number_density_hydrogen):
    """
    Saves the Dust Density to a file path given in save_path
    """
    with open(save_path, "w") as f:
        for d, density in zip(distance, total_number_density_hydrogen):
            f.write(f"{d} {density}\n")


def plot_density(distance, total_number_density_hydrogen):
    """
    Plots the Dust Density against Distance and shows in new window.
    """
    plt.plot(distance, total_number_density_hydrogen)
    plt.title("Number Density of Hydrogen (n(HI+HII))")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"n(HI + HII) $(cm^{-3})$")
    plt.show()


def convert_ebv_to_density(
    path_to_dust_density_ebv_data: str,
    save_path: str,
    save: bool = False,
    plot: bool = False,
):
    """
    Converts E(B-V) to hydrogen number density and saves or plots the data based on the arguments.

    Args:
    ----
    path_to_dust_density_ebv_data (str): Path to file with Dust Density (In E(B-V) units) and Distance data.
    save_path (str): Path to save converted dust density data.
    save (bool): Whether to save the data to a file.
    plot (bool): Whether to plot the data.
    """
    # Conversions
    kpc_in_pc = 1e3  # pc
    cm_in_m = 1e-2  # m
    pc_in_cm = 3.086e18  # cm

    # Value obtained from paper by Bohlin, Savage and Drake
    total_hydrogen_column_density_ebv_ratio = 5.8e21  # cm^-2

    # Importing the downloaded dust data
    data = np.loadtxt(path_to_dust_density_ebv_data)
    distance = data.transpose()[0]  # Distance in pc
    dust_ebv = data.transpose()[1]  # Dust in units of E(B-V) (mags)

    # Excluding the first element. First element is 0 mags for 0 pc.
    # Will cause a divide by zero error
    distance = distance[1:]
    dust_ebv = dust_ebv[1:]

    # distance = distance / kpc_in_pc # convert to kpc
    distance_in_cm = distance * pc_in_cm  # convert to cm

    # dust_ebv_per_pc = dust_ebv / distance
    dust_ebv_per_cm = dust_ebv / distance_in_cm

    total_number_density_hydrogen = (
        dust_ebv_per_cm * total_hydrogen_column_density_ebv_ratio
    )

    if save:
        save_density_to_file(save_path, distance, total_number_density_hydrogen)

    if plot:
        plot_density(distance, total_number_density_hydrogen)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")

    convert_ebv_to_density(
        os.path.join(DATA, "green-dust-ebv-2000pc.txt"),
        os.path.join(DATA, "green-dust-density-2000pc.txt"),
        save=True,
        plot=True,
    )
