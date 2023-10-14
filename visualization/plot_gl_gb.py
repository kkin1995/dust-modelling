import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

STAR = "88469"

data_path = "../data/extracted_data/fov_6_degrees/with_angle/"
df = pd.read_csv(data_path + "{}.csv".format(STAR))

glon = df["GLON"].values
glat = df["GLAT"].values
gcs = SkyCoord(glon, glat, unit = u.deg, frame = 'galactic')


plt.subplot(111, projection = 'aitoff')
plt.grid(True)
plt.scatter(gcs.l.wrap_at('180d').radian, gcs.b.radian)

plt.show()
