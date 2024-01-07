import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarWebQuery

l0, b0 = (5.9717, -1.1760)
l = np.arange(l0 - 3., l0 + 3., 0.05)
b = np.arange(b0 - 3., b0 + 3., 0.05)

l, b = np.meshgrid(l, b)

coords = SkyCoord(l * units.deg, b * units.deg, distance = 2 * units.kpc, frame = "galactic")

bayestar = BayestarWebQuery()
Av_bayestar = 2.742 * bayestar(coords)

fig = plt.figure(figsize = (4, 4), dpi = 150)
ax = fig.add_subplot()
ax.imshow(np.sqrt(Av_bayestar)[::,::-1], vmin = 0., vmax = 2., origin = "lower", interpolation = "nearest", cmap = "binary", aspect = "equal")
ax.axis("off")
ax.set_title("Centered On l = {}, b = {}".format(l0, b0))

plt.savefig("m8_dust_visualization.png", dpi = 150)
