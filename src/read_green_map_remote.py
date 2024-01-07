from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.bayestar import BayestarWebQuery
import numpy as np
import matplotlib.pyplot as plt
from dustmaps.config import config
import os

config.reset()


def query_green_dust_map(
    galactic_longitude: float,
    galactic_latitude: float,
    max_distance: float,
    plot: bool = False,
    save: bool = True,
    output_path: str = None,
):
    """
    Queries the 3D dust map from Green et al 2019 using the dustmaps package.

    Args:
    ----
    galactic_longitude (float): The galactic longitude in degrees.
    galactic_latitude (float): The galactic latitude in degrees.
    max_distance (float): The maximum distance in parsecs.
    plot (bool, optional): Whether to plot the data. Defaults to False.
    save (bool, optional): Whether to save the data to a file. Defaults to True.
    output_path (string, optional): The path to the output image or data file.

    Returns:
    ----
    tuple: A tuple containing the distances and the dust reddening in E(B-V) units.
    """

    d = np.linspace(0, max_distance, 100)

    bayestar = BayestarWebQuery(version="bayestar2019")
    coords = SkyCoord(
        galactic_longitude * u.deg,
        galactic_latitude * u.deg,
        distance=d * u.pc,
        frame="galactic",
    )

    reddening = bayestar(coords, mode="median")

    if plot:
        plt.plot(d, reddening)
        plt.title("Dust Reddening E(B-V) 2000 pc")
        plt.xlabel("Distance (pc)")
        plt.ylabel("Reddening E(B-V)")
        plt.savefig(os.path.join(output_path, "m8_bayestar2019_dust.jpg"))

    if save:
        f = open(os.path.join(output_path, "green-dust-ebv-2000pc.txt"), "w")
        for i in range(len(d)):
            f.write(str(round(d[i], 4)) + " " + str(round(reddening[i], 4)) + "\n")

        f.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")
    query_green_dust_map(5.9575, -1.1667, 2000, output_path=DATA)
