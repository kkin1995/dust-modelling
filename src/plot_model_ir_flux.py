import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_dotenv()

DATA = os.environ.get("DATA")
PLOTS = os.environ.get("PLOTS")


def plot_data(
    star_ids,
    model_name,
    path_to_observed_data_dir,
    path_to_model_dir,
    path_to_save_plots_dir,
    verbose=False,
):
    if verbose:
        print(f"Reading Model: {model_name}")
    for star in star_ids:
        if verbose:
            print(f"Star: {star}")

        model_df = pd.read_csv(
            os.path.join(path_to_model_dir, f"best_fit_{model_name}_models_{star}.csv")
        )

        angles = model_df.loc[:, "Angle"].values
        ir100 = model_df.loc[:, "IR100"].values

        # ir_obs_data = pd.read_csv(
        #     os.path.join(path_to_observed_data_dir, f"{star}_ir_100.csv")
        # )
        ir_obs_data = pd.read_csv(
            os.path.join(path_to_observed_data_dir, f"{star}_binned_ir_100.csv")
        )
        ir_obs_angles = ir_obs_data.loc[:, "Angle"]
        ir_obs_100 = ir_obs_data.loc[:, "IR100"]

        plt.plot(angles, ir100, label="Model")
        plt.scatter(
            ir_obs_angles, ir_obs_100, marker="o", color="red", s=5, label="Observed"
        )
        plt.title(f"IR Emission Model - 100 Microns - {star}")
        plt.xlabel("Angle (Degrees)")
        plt.ylabel("Flux (MJy sr-1)")
        plt.ylim(0, 2000)
        plt.legend(loc="upper right")
        plt.savefig(
            os.path.join(
                path_to_save_plots_dir,
                f"{model_name}_ir_emssion_models_binned_{star}.png",
            )
        )
        plt.clf()


if __name__ == "__main__":
    flux_data_path = os.path.join(DATA, "processed", "flux_data.csv")
    flux_data = pd.read_csv(flux_data_path)
    star_ids = flux_data.loc[:, "Star"]
    # path_to_ir_data = os.path.join(DATA, "raw", "ir_data", "extracted_data")
    path_to_ir_data = os.path.join(DATA, "derived")
    path_to_model = os.path.join(DATA, "processed")
    path_to_save_plots_dir = PLOTS

    plot_data(
        star_ids,
        "astrodust",
        path_to_ir_data,
        path_to_model,
        path_to_save_plots_dir,
        True,
    )
