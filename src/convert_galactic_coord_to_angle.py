import numpy as np
import pandas as pd
import os
from utils import setup_logger

logger = setup_logger(__name__)


def compute_angle_between_star_and_point(
    star_glon: float,
    star_glat: float,
    df_glon: float | np.ndarray,
    df_glat: float | np.ndarray,
) -> np.ndarray:
    """
    Calculates the angle between a star's location and other points in galactic coordinates, returning
    the results in degrees.

    Parameters:
    ----
    star_glon (float): Galactic Longitude of the Star.
    star_glat (float): Galactic Latitude of the Star.
    df_glon (float | np.ndarray): Galactic Longitude(s) of points around the Star.
    df_glat (float | np.ndarray): Galactic Latitude(s) of points around the Star.

    Returns:
    ----
    angles (np.ndarray): Array of angles between the coordinates of the Star and each of the points around the Star.
    """
    x0 = np.cos(np.radians(star_glat)) * np.cos(np.radians(star_glon))
    y0 = np.cos(np.radians(star_glat)) * np.sin(np.radians(star_glon))
    z0 = np.sin(np.radians(star_glat))

    x = np.cos(np.radians(df_glat)) * np.cos(np.radians(df_glon))
    y = np.cos(np.radians(df_glat)) * np.sin(np.radians(df_glon))
    z = np.sin(np.radians(df_glat))
    inner_product = np.multiply(x, x0) + np.multiply(y, y0) + np.multiply(z, z0)

    angles = np.degrees(np.arccos(inner_product))

    return angles


def convert_coordinates_for_stars(
    flux_data: pd.DataFrame,
    path_to_data_files: str,
    uv_or_ir: str,
    filename_template: str,
) -> list[pd.DataFrame]:
    """
    Converts the coordinates for stars based on provided star data and updates their corresponding data files with computed angles.

    Parameters:
    ----
    flux_data (DataFrame): DataFrame containing at least 'hip_id', 'gaia_l', and 'gaia_b' for each star.
    path_to_data_files (str): Base directory where star files are stored.
    filename_template (str): Template for filenames, expects 'hip_id' to format correctly.
    """
    if not {"hip_id", "gaia_l", "gaia_b"}.issubset(flux_data.columns):
        logger.error("Required Columns are missing from star data")
        return []

    results = []
    for _, row in flux_data.iterrows():
        filename = os.path.join(
            path_to_data_files,
            filename_template.format(hip_id=row["hip_id"], uv_or_ir=uv_or_ir),
        )
        try:
            df = pd.read_csv(filename)
            glon = df["glon"].values
            glat = df["glat"].values

            star_glon = row["gaia_l"]
            star_glat = row["gaia_b"]

            df["Angle"] = compute_angle_between_star_and_point(
                star_glon, star_glat, glon, glat
            )
            df.to_csv(filename, index=False)
            results.append(df)
        except Exception as e:

            logger.error(f"Failed to process: {e}")

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")
    uv_or_ir = "ir"
    path_to_data_files = os.path.join(DATA, "extracted_data_hlsp_files")
    flux_data_path = os.path.join(DATA, "m8_hipparcos_data_with_distance.csv")
    flux_data = pd.read_csv(flux_data_path)

    results = convert_coordinates_for_stars(
        flux_data, path_to_data_files, uv_or_ir, "hip_{hip_id}_fov_10_{uv_or_ir}.csv"
    )
