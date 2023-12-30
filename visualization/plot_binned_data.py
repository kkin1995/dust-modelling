import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

DATA = os.environ.get("DATA")
PLOTS = os.environ.get("PLOTS")

flux_data = os.path.join(DATA, "processed", "flux_data.csv")
star_ids = pd.read_csv(flux_data).loc[:, "Star"].values

binned_uv_data = os.path.join(DATA, "derived")

for star in star_ids:
    print(f"Star: {star}")
    filename = os.path.join(binned_uv_data, f"{star}_binned_nuv.csv")
    df = pd.read_csv(filename)
    angles = df.loc[:, "Angle"]
    nuv = df.loc[:, "NUV"]

    plt.scatter(angles, nuv, s=5, c="red")
    plt.xlabel("Angles (Degrees)")
    plt.ylabel("NUV (photons cm-2 s-1 A-1 sr-1)")
    plt.title(f"HIP {star}")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS, f"{star}_binned_nuv.png"))
