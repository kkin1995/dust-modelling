import os
import glob
import re
import numpy as np
from dataclasses import dataclass, field


@dataclass
class GalexData:
    """
    Data class to store information retrived from HLSP data files obtained
    from https://archive.stsci.edu/prepds/uv-bkgd/.

    Attributes:
    ----
    Name (str): Name of the GALEX pipeline file
    Date (str): Observation date from GALEX pipeline file
    Time (str): Observation time from GALEX pipeline file
    Fuv_exp_time (int): Total exposure time in the FUV band
    Nuv_exp_time (int): Total exposure time in the NUV Band
    X (int): Binned pixel in X
    Y (int): Binned pixel in Y
    Glon (float): Galactic longitude of pixel
    Glat (float): Galactic latitude of pixel
    Ecl_lon (float): Ecliptic longitude of pixel
    Ecl_lat (float): Ecliptic latitude of pixel
    Sun_ecl_lon (float): Ecliptic longitude of Sun
    Sun_ecl_lat (float): Ecliptic latitude of Sun
    Fuv_orig (int): Binned FUV flux
    Nuv_orig (int): Binned NUV flux
    Fuv_ag (int): FUV airglow contribution
    Nuv_ag (int): NUV airglow contribution
    Nuv_zl (int): NUV zodiacal light
    Fuv_final (int): Corrected diffuse astrophysical FUV
    Nuv_final (int): Corrected diffuse astrophysical NUV
    Fuv_med (int): Median FUV for this observation
    Nuv_med (int): Median NUV for this observation
    Fuv_min (int): Minimum FUV across all visits
    Nuv_min (int): Minimum NUV across all visits
    Fuv_std (int): Standard deviation across visits
    Nuv_std (int): Standard deviation across visits
    Ir100 (float): 100 micron emission
    Ebv (float): E(B-V) from Schlegel (magnitudes)
    """

    name: str = ""
    date: str = ""
    time: str = ""
    fuv_exp_time: int = 0
    nuv_exp_time: int = 0
    x: int = 0
    y: int = 0
    glon: float = 0.0
    glat: float = 0.0
    ecl_lon: float = 0.0
    ecl_lat: float = 0.0
    sun_ecl_lon: float = 0.0
    sun_ecl_lat: float = 0.0
    fuv_orig: int = 0
    nuv_orig: int = 0
    fuv_ag: int = 0
    nuv_ag: int = 0
    nuv_zl: int = 0
    fuv_final: int = 0
    nuv_final: int = 0
    fuv_med: int = 0
    nuv_med: int = 0
    fuv_min: int = 0
    nuv_min: int = 0
    fuv_std: int = 0
    nuv_std: int = 0
    ir100: float = 0.0
    ebv: float = 0.0


def extract_glat_bounds_from_filename(filename: str) -> tuple[float, float] | None:
    """
    Extracts the minimum and maximum galactic latitude from the filename of the
    HLSP files found on https://archive.stsci.edu/prepds/uv-bkgd/.

    Parameters:
    ----
    filename (str): Name of the HLSP file. Example: "hlsp_uv-bkgd_galex_diffuse_glat00-10N_fuv-nuv_v1_table.txt"

    Returns:
    ----
    glat_min, glat_max (tuple[float, float] | None): Returns the minimum and maximum galactic latitude extracted from the file name or
    returs None if the file is not named according to the above pattern.
    """
    pattern = r"glat(\d+)-(\d+)([SN])_fuv-nuv_v1_table\.txt$"
    match = re.search(pattern, filename)
    if match:
        glat_min, glat_max, hemisphere = match.groups()
        glat_min = int(glat_min)
        glat_max = int(glat_max)

        if hemisphere == "S":
            glat_min = -glat_min
            glat_max = -glat_max

        return (glat_min, glat_max)
    else:
        return None


def populate_data(data_class_instance: GalexData, values):
    """
    Populates the attributes of a GalexData instance with values from a list.

    This function dynamically assigns values to the fields of a GalexData instance
    based on the order of fields defined in the data class. It converts each value from the
    input list to the appropriate type of each field in GalexData before assignment.

    Parameters:
    ----
    data_class_instance (GalexData): An instance of GalexData to be populated.
    values (list): A list of values that correspond to the attributes of GalexData.
    The type and order of values must match the order and type requirements of GalexData fields.

    Raises:
    ----
    ValueError: If there is a type mismatch or if the number of values does not match the number of fields.

    Example:
    ----
    >>> data_instance = GalexData()
    >>> populate_data(data_instance, ['GX1', '2022-01-01', '12:00', 300, ...])
    """
    for field, value in zip(data_class_instance.__dataclass_fields__.keys(), values):
        field_type = type(getattr(data_class_instance, field))
        setattr(data_class_instance, field, field_type(value))


def calculate_summary(
    n_lines: int, fuv_total: float, fuv_time: float, nuv_total: float, nuv_time: float
) -> np.ndarray:
    """
    Calculate summary statistics for output data in array `out`.

    Parameters:
    ----
    n_lines (int): Number of lines in HLSP file which are not comments or empty.
    fuv_total (float): Total FUV exposure for field of view.
    fuv_time (float): Total FUV exposure time.
    nuv_total (float): Total NUV exposure for field of view.
    nuv_time (float): Total NUV exposure time.

    Returns:
    ----
    out_sum (np.ndarray): Array containing number of lines of data, weighted FUV, total FUV exposure time,
    weighted NUV, and total NUV exposure time.
    """
    out_sum = np.zeros(5)
    out_sum[0] = n_lines
    out_sum[1] = -9999
    out_sum[3] = -9999
    if fuv_time > 0:
        out_sum[1] = fuv_total / fuv_time
        out_sum[2] = fuv_time
    if nuv_time > 0:
        out_sum[3] = nuv_total / nuv_time
        out_sum[4] = nuv_time
    return out_sum


def galactic_to_cartesian(gl: float, gb: float) -> tuple[float, float, float]:
    """
    Convertes a pair of galactic (spherical) coordinates to cartesian coordinates.

    Parameters:
    ----
    gl (float): Galactic Longitude.
    gb (float): Galactic Latitude.

    Returns:
    ----
    x, y, z (tuple[float, float, float]): Cartesian coordinates.
    """
    x = np.cos(np.radians(gl)) * np.cos(np.radians(gb))
    y = np.sin(np.radians(gl)) * np.cos(np.radians(gb))
    z = np.sin(np.radians(gb))
    return x, y, z


def is_within_fov(
    data: GalexData, x: float, y: float, z: float, cos_fov: float
) -> bool:
    """
    Checks if the galactic coordinates from the `GalexData` instance is within the field of view.

    Parameters:
    ----
    data (GalexData): Instance of GalexData containinng each line of data from HLSP files.
    x (float): The x coordinate of specified point in the center of the field of view.
    y (float): The y coordinate of specified point in the center of the field of view.
    z (float): The z coordinate of specified point in the center of the field of view.
    cos_fov (float): The cosine of the field of view specified.

    Returns:
    ----
    bool: True if data point is within field of view, False otherwise.
    """
    x1, y1, z1 = galactic_to_cartesian(data.glon, data.glat)
    return (x1 * x + y1 * y + z1 * z) > cos_fov


def update_totals(
    data: GalexData, fuv_total: int, fuv_time: int, nuv_total: int, nuv_time: int
) -> tuple[int, int, int, int]:
    """
    Updates the total and time counters for FUV and NUV data.

    Parameters:
    ----
    data (GalexData): Instance of GalexData containinng each line of data from HLSP files.
    fuv_total (int): Running total of the FUV data.
    fuv_time (int): Running total of the FUV exposure time.
    nuv_total (int): Running total of the NUV data.
    nuv_time (int): Running total of the NUV exposure time.

    Returns:
    ----
    fuv_total, fuv_time, nuv_total, nuv_time (tuple[int, int, int, int]): Updated data.
    """
    if (data.fuv_exp_time > 0) and (data.fuv_final >= 0):
        fuv_total += data.fuv_final * data.fuv_exp_time
        fuv_time += data.fuv_exp_time
    if (data.nuv_exp_time > 0) and (data.nuv_final >= 0):
        nuv_total += data.nuv_final * data.nuv_exp_time
        nuv_time += data.nuv_exp_time

    return fuv_total, fuv_time, nuv_total, nuv_time


def file_in_fov(filename: str, min_gb: float, max_gb: float) -> bool:
    """
    Checks if file is witin field of view based on metadata available in the filename.

    Parameters:
    ----
    filename (str): Name of the HLSP file. Example: "hlsp_uv-bkgd_galex_diffuse_glat00-10N_fuv-nuv_v1_table.txt".
    min_gb (float): Minimum value of the galactic latitude calculated from the field of view.
    max_gb (float): Maximum value of the galactic latitude calculated from the field of view.

    Returns:
    ----
    bool: True if file is witin field of view, False otherwise.
    """
    glat_min, glat_max = extract_glat_bounds_from_filename(filename)
    return (min_gb <= glat_max) and (max_gb >= glat_min)


def check_valid_filenames_in_directory(filenames: list[str]) -> list[str]:
    """
    Checks if filenames follow the pattern: `glat(\d+)-(\d+)([SN])_fuv-nuv_v1_table\.txt$`.

    Parameters:
    ----
    filenames (list[str]): List of filenames to check.

    Returns:
    ----
    valid_filenames (list[str]): List of filenames that follow the pattern.
    """
    pattern = r"glat(\d+)-(\d+)([SN])_fuv-nuv_v1_table\.txt$"

    # Filter files to match the regex pattern
    valid_filenames = [
        filename
        for filename in filenames
        if re.search(pattern, os.path.basename(filename))
    ]
    return valid_filenames


def get_galex_hlsp_data(
    gl: float, gb: float, fov: float, data_dir: str, verbose: bool = False
) -> tuple[list[GalexData], np.ndarray]:
    """
    Retrieves and processes GALEX HLSP data files within a specified field of view from given galactic coordinates.

    Parameters:
    ----
    gl (float): Galactic longitude in degrees.
    gb (float): Galactic latitude in degrees.
    fov (float): Field of view in degrees.
    data_dir (str): Directory containing data (HLSP) files. The directory should contain .txt files.
    Example File Name: "hlsp_uv-bkgd_galex_diffuse_glat00-10N_fuv-nuv_v1_table.txt"
    verbose (bool, optional): Verbosity Flag.

    Returns:
    ----
    tuple: A tuple containing:
        - list of GalexData instances with data within the specified FOV.
        - numpy array containing summary data (total number of rows, weighted FUV, total FUV time,
            weighted NUV, total NUV time).

    Raises:
    ----
    IOError: If files in the directory cannot be read.
    ValueError: If data extraction from filenames or file content fails.
    """
    if data_dir is None:
        raise ValueError("Data directory must be specified.")

    out = []

    filenames = glob.glob(os.path.join(data_dir, "*.txt"))

    valid_filenames = check_valid_filenames_in_directory(filenames)
    if not valid_filenames:
        raise ValueError(
            "No valid files found in the directory that match the required pattern."
        )

    fuv_total = 0
    nuv_total = 0
    fuv_time = 0
    nuv_time = 0

    # Minimum and maximum galactic latitude.
    min_gb = gb - fov
    max_gb = gb + fov

    # Convert Galactic Coordinates to Cartesian
    x, y, z = galactic_to_cartesian(gl, gb)

    cos_fov = np.cos(np.radians(fov))

    files_to_read = []
    for _, filename in enumerate(filenames):
        if file_in_fov(filename, min_gb, max_gb):
            files_to_read.append(filename)
    if verbose:
        print(f"Found {len(files_to_read)} files with {fov} degrees field of view.")
        print("----------------------------------------------------------------")

    n_lines = 0
    for idx, filename in enumerate(files_to_read):
        if verbose:
            print(f"Reading File Number: {idx} with Name: {filename}")
            print("----------------------------------------------------------------")
        with open(filename, "r") as f:
            for line in f:
                data = GalexData()
                # Skipping Comment Lines (Starts With #) and empty lines
                if line.strip() and not line.startswith("#"):
                    n_lines += 1
                    if n_lines % 10000 == 0 and verbose:
                        print(f"Reading Line: {n_lines}")

                    words = line.split()
                    populate_data(data, words)

                    if is_within_fov(data, x, y, z, cos_fov):
                        out.append(data)
                        fuv_total, fuv_time, nuv_total, nuv_time = update_totals(
                            data, fuv_total, fuv_time, nuv_total, nuv_time
                        )

    out_sum = calculate_summary(n_lines, fuv_total, fuv_time, nuv_total, nuv_time)

    return out, out_sum


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import pandas as pd
    from dataclasses import asdict

    load_dotenv()
    DATA = os.environ.get("DATA")
    star_id = 88496
    galex_hlsp_dir = os.path.join(DATA, "galex_hlsp_files/")
    m8_star_data_df = pd.read_csv(
        os.path.join(DATA, "m8_hipparcos_data_with_distance.csv")
    )
    star_data = m8_star_data_df.loc[m8_star_data_df.loc[:, "hip_id"] == star_id, :]
    gl = star_data["gaia_l"].values[0]
    gb = star_data["gaia_b"].values[0]
    fov = 10

    out, out_sum = get_galex_hlsp_data(gl, gb, fov, galex_hlsp_dir, verbose=True)
    data_dicts = [asdict(instance) for instance in out]
    df = pd.DataFrame(data_dicts)
    df.to_csv(
        os.path.join(
            DATA, "extracted_data_hlsp_files", f"hip_{str(star_id)}_fov_{fov}.csv"
        )
    )
    print(out_sum)
