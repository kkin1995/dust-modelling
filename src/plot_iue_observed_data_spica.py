import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from dotenv import load_dotenv

load_dotenv()
DATA = os.environ.get("DATA")

iue_spica_data_path = os.path.join(
    DATA, "processed", "iue-data-spica-six-degrees-with-flux-at-1500-A.csv"
)
spica_hipparcos_data_path = os.path.join(
    DATA, "raw", "spica_hipparcos_data_with_distance.csv"
)
spica_hippacos_data = pd.read_csv(spica_hipparcos_data_path)
spica_ra = spica_hippacos_data.loc[:, "ra"]
spica_dec = spica_hippacos_data.loc[:, "dec"]

coords = SkyCoord(spica_ra, spica_dec, unit=u.deg, frame="icrs")
gcs = coords.galactic
gl0 = gcs.l.degree
gb0 = gcs.b.degree

x0 = np.cos(np.radians(gb0)) * np.cos(np.radians(gl0))
y0 = np.cos(np.radians(gb0)) * np.sin(np.radians(gl0))
z0 = np.sin(np.radians(gb0))

df = pd.read_csv(iue_spica_data_path, skiprows=1, index_col=0)
df.columns = [
    "Data ID",
    "GO Object Name",
    "Target Name",
    "RA (J2000)",
    "Dec (J2000)",
    "Exp Time",
    "Disp",
    "Aper",
    "Category",
    "Obs Start Time",
    "Program ID",
    "High-Level Science Products",
    "Ref",
    "Ang Sep (')",
    "gl",
    "gb",
    "WAVELENGTH",
    "FLUX(ERGCM-2S-1A-1)",
]
gl = df.loc[:, "gl"].values
gb = df.loc[:, "gb"].values

x = np.cos(np.radians(gb)) * np.cos(np.radians(gl))
y = np.cos(np.radians(gb)) * np.sin(np.radians(gl))
z = np.sin(np.radians(gb))

inner_product = np.multiply(x, x0) + np.multiply(y, y0) + np.multiply(z, z0)
angles = np.degrees(np.arccos(inner_product))

flux = df.loc[:, "FLUX(ERGCM-2S-1A-1)"]

plt.scatter(angles, flux, s=2, c="red")
plt.show()
