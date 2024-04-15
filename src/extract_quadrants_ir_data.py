import pandas as pd
import os
from utils import setup_logger

logger = setup_logger(__name__)


def extract_quadrants(
    star_list: list,
    ir_data_file_paths: list,
    path_to_star_data: str,
    path_to_save_extracted_quadrants: str,
):
    """
    Extracts infrared (IR) data into four quadrants (north, south, east, west) based on each star's galactic coordinates and saves them to specified locations.
    This function iterates over a list of stars, locates their IR data files, and reads the star's data to determine its galactic longitude and latitude. It then divides the IR data into quadrants relative to the star's galactic coordinates and saves the quadrant data into separate files.

    Parameters:
    ----
    star_list (list): A list of star identifiers as strings. Each identifier is used to find corresponding IR data files.
    ir_data_file_paths (list): A list of file paths where each star's IR data is stored.
    path_to_star_data (str): The file path where data about each star (including their galactic coordinates) is stored. This file should contain at least 'hip_id', 'gaia_l', and 'gaia_b' columns.
    path_to_save_extracted_quadrants (str): The directory path where the quadrant data files will be saved. Each quadrant file will be named according to the star and the quadrant (e.g., '88469_ir_with_angles_north.csv').

    Raises:
    ----
    FileNotFoundError:
        If the file for a given star is not found in `ir_data_file_paths`.
    Exception:
        Catches any other exceptions that may occur during the reading of IR data files or processing of data.

    Notes:
    ----
    This function requires the presence of specific columns ('gaia_l', 'gaia_b' for galactic coordinates) in the star data CSV file.
    It assumes that the IR data files are named in a manner that includes the star's identifier.
    """
    for star in star_list:
        logger.info(f"Extracting Quadrants For Star: {star}")
        ir_data_filename = next(
            (file_path for file_path in ir_data_file_paths if star in file_path), None
        )
        try:
            ir_data = pd.read_csv(ir_data_filename)
            star_data = pd.read_csv(path_to_star_data)
            star_data = star_data[star_data["hip_id"] == int(star)]

            if star_data.empty:
                logger.warning(f"No data found for star {star}")
                continue

            star_gl = star_data["gaia_l"].values[0]
            star_gb = star_data["gaia_b"].values[0]

            ir_data_north = ir_data[ir_data["GB"] >= star_gb]
            ir_data_south = ir_data[ir_data["GB"] <= star_gb]
            ir_data_west = ir_data[ir_data["GL"] <= star_gl]
            ir_data_east = ir_data[ir_data["GL"] >= star_gl]

            save_paths = {
                "north": os.path.join(
                    path_to_save_extracted_quadrants, star, "_ir_with_angles_north.csv"
                ),
                "south": os.path.join(
                    path_to_save_extracted_quadrants, star, "_ir_with_angles_south.csv"
                ),
                "west": os.path.join(
                    path_to_save_extracted_quadrants, star, "_ir_with_angles_west.csv"
                ),
                "east": os.path.join(
                    path_to_save_extracted_quadrants, star, "_ir_with_angles_east.csv"
                ),
            }
            ir_data_north.to_csv(
                save_paths["north"],
                index=False,
            )
            ir_data_south.to_csv(
                save_paths["south"],
                index=False,
            )
            ir_data_west.to_csv(
                save_paths["west"],
                index=False,
            )
            ir_data_east.to_csv(
                save_paths["east"],
                index=False,
            )
        except Exception as e:
            logger.error(f"Failed to extract IR data into quadrants. Error: {e}")


if __name__ == "__main__":
    from utils import get_file_paths

    star_list = [
        "88469",
        "88496",
        "88506",
        "88380",
        "88581",
        "88560",
        "88256",
        "88142",
        "88705",
        "88463",
    ]
    path_to_star_data = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/data/m8-stellar-data/m8_stellar_data_gaia_hipparcos_with_computed_distance.csv"
    path_to_save_extracted_quadrants = "/Users/karankinariwala/Library/CloudStorage/Dropbox/KARAN/1-College/MSc/4th-Semester/dust_modeling/data/extracted_ir_data/with_angle"
    path_to_ir_data = "/Users/karankinariwala/Library/CloudStorage/Dropbox/KARAN/1-College/MSc/4th-Semester/dust_modeling/data/extracted_ir_data/with_angle"
    ir_data_file_paths = get_file_paths(path_to_ir_data)

    extract_quadrants(
        star_list,
        ir_data_file_paths,
        path_to_star_data,
        path_to_save_extracted_quadrants,
    )
