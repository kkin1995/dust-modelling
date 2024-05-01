import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import os


def plot_data(star_id, data_dir, plot_dir, data_type, wavelength=None, bin_size=0.03):
    """
    Plots data for a given star based on the specified data type (UV or IR).

    Parameters:
    ----
    star_id (int): Identifier for the star.
    data_dir (str): Directory containing the data files.
    plot_dir (str): Directory to save the plots.
    data_type (str): Type of data ('uv' or 'ir').
    wavelength (str, optional): Wavelength type ('fuv' or 'nuv') if data_type is 'uv'.
    bin_size (float): The bin size used in the data.
    """
    if data_type == "uv" and wavelength is None:
        raise ValueError("Wavelength must be specified for UV data ('fuv' or 'nuv')")

    y_axes = f"{wavelength}" if data_type == "uv" else "ir100"
    filename = f"hip_{star_id}_fov_10_{data_type}_{'' if data_type == 'ir' else wavelength + '_'}binned_{bin_size}.csv"
    file_path = os.path.join(data_dir, filename)

    print(
        f"Plotting for Star: {star_id}, Data Type: {data_type.upper()}, Wavelength: {wavelength}"
    )

    try:
        df = pd.read_csv(file_path)
        plt.figure(figsize=(10, 6))
        plt.scatter(df["Angle"], df[y_axes], s=5, color="red")
        plt.xlabel("Angle (Degrees)")
        plt.ylabel(f"{data_type.upper()}")
        plt.title(f"HIP {star_id} - Binned {data_type.upper()}")

        output_path = os.path.join(plot_dir, f"{filename}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error while plotting data for star {star_id}: {e}")


if __name__ == "__main__":
    load_dotenv()
    DATA = os.environ.get("DATA")

    flux_data = os.path.join(DATA, "flux_data_m8.csv")
    star_ids = [88469, 88496]
    binned_data_dir = os.path.join(DATA, "extracted_data_hlsp_files")
    plot_dir = os.path.join(DATA, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Example of plotting IR and UV data for each star
    for star in star_ids:
        plot_data(star, binned_data_dir, plot_dir, "ir")
        plot_data(star, binned_data_dir, plot_dir, "uv", "fuv")
        plot_data(star, binned_data_dir, plot_dir, "uv", "nuv")
