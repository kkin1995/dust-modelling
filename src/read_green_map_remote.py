from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.bayestar import BayestarWebQuery
import numpy as np
import matplotlib.pyplot as plt
from dustmaps.config import config
import os
from utils import setup_logger

logger = setup_logger(__name__)

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
    output_path (string, optional): The path to the output data file.

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
        f = open(output_path, "w")
        f.write("Distance(pc)" + " " + "Dust(EBV)" + "\n")
        for i in range(len(d)):
            f.write(str(round(d[i], 4)) + " " + str(round(reddening[i], 4)) + "\n")

        f.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()
    DATA = os.environ.get("DATA")

    m8_star_data = pd.read_csv(
        os.path.join(DATA, "m8_hipparcos_data_with_distance.csv")
    )
    output_data_path = os.path.join(DATA, "dust_data_green_2019/")
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    for idx, row in m8_star_data.iterrows():
        star_id = row["hip_id"]
        star_gl = row["gaia_l"]
        star_gb = row["gaia_b"]
        star_distance = row["Distance(pc)"]
        logger.info(
            f"Querying Dust Map for {star_id} at coordinates ({star_gl}, {star_gb})"
        )
        query_green_dust_map(
            star_gl,
            star_gb,
            star_distance,
            output_path=os.path.join(output_data_path, f"{star_id}.csv"),
        )
