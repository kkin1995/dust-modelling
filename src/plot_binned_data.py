import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

DATA = os.environ.get("DATA")

flux_data = os.path.join(DATA, "flux_data_m8.csv")
star_ids = pd.read_csv(flux_data).loc[:, "Star"].values

binned_uv_data = os.path.join(
    DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/binned_nuv_data"
)
binned_ir_data = os.path.join(
    DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/binned_ir_data"
)

for star in star_ids:
    print(f"Star: {star}")
    filename = os.path.join(binned_ir_data, f"{star}_binned_ir.csv")
    df = pd.read_csv(filename)
    angles = df.loc[:, "Angle"]
    ir = df.loc[:, "IR100"]

    plt.scatter(angles, ir, s=5, c="red")
    plt.xlabel("Angles (Degrees)")
    plt.ylabel("IR Emission (MJy sr-1)")
    plt.title(f"HIP {star} - Binned IR Emission")
    plt.grid(True)
    plt.savefig(os.path.join(DATA, "plots", f"{star}_binned_ir.png"))
