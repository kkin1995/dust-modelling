import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
DATA = os.environ.get("DATA")

star = "65474"

df = pd.read_csv(os.path.join(DATA, "processed", "flux_angle_model_65474.csv"))
# df = pd.read_csv(f"{DATA}euv_angle_model_{star}.csv")
angles = df.loc[:, "Angle"]
flux = df.loc[:, "Flux"]

plt.plot(angles, flux)
plt.title(f"HIP {star} Inverse Square Model")
plt.xlabel("Angle")
plt.ylabel("EUV")
plt.show()
