from model_ir_emission_astrodust import IREmissionModeler
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
DATA = os.environ.get("DATA")

star_id = 88581

path_to_stellar_model_flux = os.path.join(DATA, "processed", "flux_data.csv")
path_to_astrodust_model = os.path.join(
    DATA, "raw", "ir_data", "astrodust+PAH_MW_RV3.1.fits"
)
path_to_dust_density_file = os.path.join(
    DATA, "processed", "green-dust-density-2000pc.txt"
)
path_to_ir_data_dir = os.path.join(DATA, "raw", "ir_data", "extracted_data")
path_to_binned_ir_data_dir = os.path.join(DATA, "derived")

path_to_col_density_file = os.path.join(DATA, "processed", "m8_col_density.yaml")

modeler = IREmissionModeler(
    star_id,
    [0, 0],
    path_to_stellar_model_flux,
    path_to_astrodust_model,
    path_to_dust_density_file,
    path_to_ir_data_dir,
    path_to_binned_ir_data_dir,
    path_to_col_density_file,
)

a = np.arange(0.0, 1.0, 0.1)
g = np.arange(0.0, 0.999, 0.1)

num_points = a.shape[0]

A, G = np.meshgrid(a, g)

chi_sq_grid = np.zeros_like(A)

for ia in range(num_points):
    for ig in range(num_points):
        a_val = A[ia, ig]
        g_val = G[ia, ig]

        chi_sq = modeler.fit([a_val, g_val], modeler.sflux, modeler.dstar)
        chi_sq_grid[ia, ig] = chi_sq

plt.figure(figsize=(8, 6))
contour_levels = np.linspace(chi_sq_grid.min(), chi_sq_grid.max(), 50)
plt.contourf(A, G, chi_sq_grid, levels=contour_levels)
plt.colorbar()
plt.xlabel("Albedo (a)")
plt.ylabel("Phase (g)")
plt.title("Chi Square Values over Parameter Grid")
plt.savefig(f"{star_id}_chi_square_params_grid.png")
