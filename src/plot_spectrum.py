import os
from dotenv import load_dotenv
from star_model import StarModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

load_dotenv()
DATA = os.environ.get("DATA")

path_to_model_spectra = os.path.join(DATA, "castelli-kurucz-models")
star_data = pd.read_csv(os.path.join(DATA, "m8_stellar_data_gaia_hipparcos.csv"))
stars = star_data.loc[:, "hip_id"]
wave = np.linspace(1000, 11000, num=10000)

for star in stars:
    sflux = np.load(os.path.join(path_to_model_spectra, f"{star}.npy"))

    plt.figure(figsize=(10, 7))
    plt.plot(wave, sflux)

    plt.grid(True)
    plt.title(f"HIP {star} Model Spectrum")
    plt.xlabel("Wavelength (Angstroms)")
    plt.ylabel("Flux (egrs cm-2 s-1 A-1)")
    plt.savefig(
        os.path.join(path_to_model_spectra, "model_spectra_figures", f"{star}.png")
    )
