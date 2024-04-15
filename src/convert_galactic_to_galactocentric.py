import astropy.units as u
import astropy.coordinates as coord
import pandas as pd
from utils import setup_logger

logger = setup_logger(__name__)


def convert_to_galactocentric(galactic_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Galactic coordinates to Galactocentric coordinates.

    Parameters:
    ----
    galactic_coords (pd.DataFrame): DataFrame containing stellar data with columns "Distance(pc)", "gaia_l", and "gaia_b".

    Returns:
    ----
    galactic_coords (pd.DataFrame): DataFrame with Galactocentric coordinates x, y, and z added.
    """

    required_columns = ["Distance(pc)", "gaia_l", "gaia_b"]
    if not all(column in galactic_coords.columns for column in required_columns):
        logger.error(ValueError("DataFrame is missing one or more required columns."))
        raise ValueError("DataFrame is missing one or more required columns.")

    try:
        distance_pc = galactic_coords["Distance(pc)"].values
        gl = galactic_coords["gaia_l"].values
        gb = galactic_coords["gaia_b"].values
    except KeyError as e:
        logger.error(KeyError(f"Missing Column in DataFrame: {e}"))
        raise KeyError(f"Missing Column in DataFrame: {e}")
    except ValueError as e:
        logger.error(ValueError(f"Data Conversion Error: {e}"))
        raise ValueError(f"Data Conversion Error: {e}")

    try:
        c = coord.SkyCoord(
            l=gl * u.deg, b=gb * u.deg, distance=distance_pc * u.pc, frame="galactic"
        )

        c_galactocentric = c.transform_to(coord.Galactocentric)
    except Exception as e:
        logger.error(f"Error from Astropy: {e}")
        raise e

    galactic_coords["x_galactocentric"] = c_galactocentric.x.value
    galactic_coords["y_galactocentric"] = c_galactocentric.y.value
    galactic_coords["z_galactocentric"] = c_galactocentric.z.value

    return galactic_coords


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    DATA = os.environ.get("DATA")

    df = pd.read_csv(os.path.join(DATA, "m8_hipparcos_data_with_distance.csv"))

    df_with_galactocentric = convert_to_galactocentric(df)

    # Save or print the new DataFrame as required
    print(df_with_galactocentric)
