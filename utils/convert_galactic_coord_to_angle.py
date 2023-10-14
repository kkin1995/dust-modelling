import numpy as np
import pandas as pd
import os


def compute_angle_between_star_and_point(star_glon, star_glat, df_glon, df_glat):
    """
    Calculates the angles between a star's location and other points in galactic coordinates.

    Args:
    ----
    star_glon (float): Galactic Longitude of the Star.
    star_glat (float): Galactic Latitude of the Star.
    df_glon (float | np.ndarray): Galactic Longitude(s) of points around the Star.
    df_glat (float | np.ndarray): Galactic Latitude(s) of points around the Star.
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


def convert_coordinates_for_stars(path_to_flux_data, path_to_ir_data_dir):
    """
    Converts the coordinates for a specific star and saves the results.
    """
    flux_data = pd.read_csv(path_to_flux_data)
    stars = flux_data.loc[:, "Star"]
    for star in stars:
        print(f"Star: {star}")
        filename = os.path.join(path_to_ir_data_dir, f"{star}_ir_100.csv")
        df = pd.read_csv(filename)
        star_glon = flux_data[flux_data["hip_id"] == star]["gaia_l"].values
        star_glat = flux_data[flux_data["hip_id"] == star]["gaia_b"].values

        glon = df["GL"].values
        glat = df["GB"].values

        df["Angle"] = compute_angle_between_star_and_point(
            star_glon, star_glat, glon, glat
        )

    return df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")
    ir_data_dir = os.path.join(DATA, "raw", "ir_data", "extracted_data")
    processed_data_dir = os.path.join(DATA, "processed")
    flux_data_path = os.path.join(processed_data_dir, "flux_data.csv")
    m8_star_data_path = os.path.join(
        processed_data_dir, "m8_stellar_data_gaia_hipparcos_with_computed_distance.csv"
    )
