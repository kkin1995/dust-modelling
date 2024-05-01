from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
DATA = os.environ.get("DATA")

path_to_observed_data = os.path.join(
    DATA, "extracted_data_hlsp_files", "csv", "fov_6_degrees"
)
path_to_star_data = os.path.join(DATA, "m8_hipparcos_data_with_distance.csv")

star_data = pd.read_csv(path_to_star_data)
star_ids = star_data["hip_id"]

for star_id in star_ids:
    print(f"Calculating Density Map for {star_id}")

    star_gl = star_data.loc[star_data.loc[:, "hip_id"] == star_id, "gaia_l"]
    star_gb = star_data.loc[star_data.loc[:, "hip_id"] == star_id, "gaia_b"]

    filepath = os.path.join(path_to_observed_data, f"extracted_observed_{star_id}.csv")
    df = pd.read_csv(filepath)
    # For IR Data
    gl = df["GLON"]
    gb = df["GLAT"]
    gc = np.vstack([gl, gb])
    kde = gaussian_kde(gc)

    l_grid = np.linspace(np.min(gl), np.max(gl), 100)
    b_grid = np.linspace(np.min(gb), np.max(gb), 100)
    l_grid, b_grid = np.meshgrid(l_grid, b_grid)
    data_grid = np.vstack([l_grid.ravel(), b_grid.ravel()])

    density = kde(data_grid).reshape(l_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.rot90(density),
        extent=[np.min(gl), np.max(gl), np.min(gb), np.max(gb)],
        cmap="Blues",
    )
    plt.colorbar(label="density")
    plt.scatter(star_gl, star_gb, color="red", s=5, label=f"HIP {star_id}")
    # plt.scatter(gl, gb, color="red", s=0.5)
    plt.legend()
    plt.xlabel("Galactic Longitude")
    plt.ylabel("Galactic Latitude")
    plt.title(f"HIP {star_id} - Density of IR Data Points")
    plt.savefig(os.path.join(DATA, "plots", f"{star_id}_ir_density_plot.jpg"))
