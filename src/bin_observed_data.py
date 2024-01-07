import pandas as pd
import numpy as np


def bin_data(
    data_file: str,
    bin_size: float,
    x_data_name: str,
    y_data_name: str,
    uv_or_ir="ir",
) -> tuple[np.array, np.array]:
    """
    This function reads data from a CSV file and sorts it based on a given column name. It then bins the data
    based on a specified bin size and calculates the mean y value for each bin.

    Parameters:
    data_file (str): Path to the input CSV file.
    bin_size (float): The size of the bin to split the x data into.
    x_data_name (str): The name of the column in the CSV file to use as x data.
    y_data_name (str): The name of the column in the CSV file to use as y data.

    Returns:
    np.array: An array of the mean x values for each bin.
    np.array: An array of the mean y values for each bin.
    """
    df = pd.read_csv(data_file)

    if uv_or_ir == "uv":
        df.drop(df.loc[df["NUV"] == -9999].index, inplace=True)
        df.drop(df.loc[df["NUV_STD"] == 1000000].index, inplace=True)
        # df.drop(df.loc[df["FUV"] == -9999].index, inplace=True)
        # df.drop(df.loc[df["FUV_STD"] == 1000000].index, inplace=True)

    df.sort_values(by=x_data_name, axis=0, inplace=True)

    x_data = df.loc[:, x_data_name].values
    y_data = df.loc[:, y_data_name].values

    max_x = max(x_data)

    binned_x = []
    binned_y = []

    i = 0
    start_of_bin = x_data[i]
    end_of_bin = start_of_bin + bin_size
    while end_of_bin <= max_x:
        summed_y = 0
        no_of_y = 0
        binned_x.append((start_of_bin + end_of_bin) / 2.0)

        while i < len(x_data) and x_data[i] < end_of_bin:
            summed_y += y_data[i]
            no_of_y += 1
            i += 1

        if no_of_y == 0:
            raise ValueError(
                "Bin Size too small. No points exist inside bin. Consider increasing bin size."
            )

        binned_y.append(summed_y / no_of_y)
        start_of_bin = end_of_bin
        end_of_bin += bin_size

    return np.array(binned_x), np.array(binned_y)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    DATA = os.environ.get("DATA")

    ir_observed_data_dir = os.path.join(
        DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/with_angle/"
    )
    flux_data = os.path.join(DATA, "flux_data_m8.csv")
    star_ids = pd.read_csv(flux_data).loc[:, "Star"].values

    bin_size = 0.03

    for star_id in star_ids:
        filename = os.path.join(ir_observed_data_dir, f"{star_id}.csv")
        print(f"File: {filename}")

        try:
            binned_angles, binned_flux = bin_data(filename, bin_size, "Angle", "IR100")
        except Exception as e:
            print(e)
            continue

        df = pd.DataFrame(data={"Angle": binned_angles, "IR100": binned_flux})
        df.to_csv(
            os.path.join(
                DATA,
                os.path.join(
                    "extracted_data_hlsp_files/csv/fov_6_degrees/binned_ir_data",
                    f"{star_id}_binned_ir.csv",
                ),
            )
        )
