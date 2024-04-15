import pandas as pd
import numpy as np
from utils import setup_logger

logger = setup_logger(__name__)


def bin_data(
    data_file: str,
    bin_size: float,
    x_data_name: str,
    y_data_name: str,
    uv_or_ir="ir",
    fuv_or_nuv: str = "fuv",
) -> tuple[np.array, np.array]:
    """
    This function reads data from a CSV file and sorts it based on a given column name. It then bins the data
    based on a specified bin size and calculates the mean y value for each bin.

    Parameters:
    ----
    data_file (str): Path to the input CSV file.
    bin_size (float): The size of the bin to split the x data into.
    x_data_name (str): Column name in the CSV file to use as x data.
    y_data_name (str): Column name in the CSV file to use as y data. For UV data, CSV file must have columns
    FUV, NUV, FUV_STD, NUV_STD.
    uv_or_ir (str): Specify 'uv' for ultraviolet data, otherwise 'ir' for infrared.
    fuv_or_nuv (str): Specify 'fuv' for Far UV data, otherwise 'nuv' for Near UV data.

    Returns:
    ----
    tuple[np.ndarray, np.ndarray]: Tuple containing arrays of the mean x values and mean y
    values for each bin.
    """
    try:
        df = pd.read_csv(data_file)

        if uv_or_ir == "uv":
            key = f"{fuv_or_nuv.upper()}"
            key_std = f"{key}_STD"
            df = df[df[key] != -9999]
            df = df[df[key_std] != 1000000]

        df.sort_values(by=x_data_name, axis=0, inplace=True)

        bins = pd.interval_range(
            start=df[x_data_name].min(),
            end=df[x_data_name].max(),
            freq=bin_size,
            closed="right",
        )

        df["bins"] = pd.cut(df[x_data_name], bins=bins)
        binned_data = (
            df.groupby("bins").agg({x_data_name: "mean", y_data_name: "mean"}).dropna()
        )

        return binned_data[x_data_name].values, binned_data[y_data_name].values

    except pd.errors.EmptyDataError:
        logger.error("Data file is empty or doesn't exist")
        return np.array([]), np.array([])
    except KeyError as e:
        logger.error(f"Missing expected column in data: {e}")
        return np.array([]), np.array([])
    except Exception as e:
        logger.error(f"An error occured: {e}")
        return np.array([]), np.array([])


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    DATA = os.environ.get("DATA")

    data_dir = os.path.join(
        DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/with_angle/"
    )

    flux_data = os.path.join(DATA, "flux_data_m8.csv")
    star_ids = pd.read_csv(flux_data).loc[:, "Star"].values

    bin_size = 0.03

    for star_id in star_ids:
        filename = os.path.join(data_dir, f"{star_id}.csv")
        logger.info(f"Binning Data For {star_id}")

        binned_angles, binned_flux = bin_data(
            filename, bin_size, "Angle", "NUV", "uv", "nuv"
        )

        df = pd.DataFrame(data={"Angle": binned_angles, "NUV": binned_flux})
        df.to_csv(
            os.path.join(
                DATA,
                os.path.join(
                    "extracted_data_hlsp_files/csv/fov_6_degrees/binned_nuv_data",
                    f"{star_id}_binned_nuv.csv",
                ),
            )
        )
