from model_ir_emission_astrodust import IREmissionModeler
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from utils import setup_logger

logger = setup_logger(__name__)

load_dotenv()
DATA = os.environ.get("DATA")


class ChiSquareGrid:

    def __init__(self, star_id, config):
        self.star_id = star_id
        self.config = config
        self.setup_modeler()

    def setup_modeler(self):
        self.modeler = IREmissionModeler(self.star_id, [0, 0], self.config)
        self.a_values = np.arange(0.0, 1.0, 0.1)
        self.g_values = np.arange(0.0, 0.999, 0.1)

    def compute_chi_square_grid(self):
        num_points = self.a_values.shape[0]
        A, G = np.meshgrid(self.a_values, self.g_values)
        chi_sq_grid = np.zeros_like(A)

        for ia in range(num_points):
            print(f"Albedo: {self.a_values[ia]:.2f}")
            for ig in range(num_points):
                print(f"Phase: {self.g_values[ig]:.2f}")
                chi_sq = self.modeler.fit(
                    [A[ia, ig], G[ia, ig]], self.modeler.sflux, self.modeler.dstar
                )
                chi_sq_grid[ia, ig] = chi_sq

        return A, G, chi_sq_grid

    def plot_chi_square_grid(self, A, G, chi_sq_grid):
        plt.figure(figsize=(6, 6))
        contour_levels = np.linspace(chi_sq_grid.min(), chi_sq_grid.max(), 50)
        plt.contourf(A, G, chi_sq_grid, levels=contour_levels)
        plt.colorbar()
        plt.xlabel("Albedo (a)")
        plt.ylabel("Phase (g)")
        plt.title(f"HIP {self.star_id} Chi Square Values Over Parameter Grid")
        plt.savefig(f"{self.star_id}_chi_square_params_grid.png")
        plt.close()


if __name__ == "__main__":
    star_id = 88469

    config = {
        "path_to_stellar_model_flux": os.path.join(DATA, "flux_data_m8.csv"),
        "path_to_astrodust_model": os.path.join(DATA, "astrodust+PAH_MW_RV3.1.fits"),
        "path_to_dust_density_file": os.path.join(
            DATA, "green-dust-density-2000pc.txt"
        ),
        "path_to_ir_data_dir": os.path.join(
            DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/with_angle"
        ),
        "path_to_binned_ir_data_dir": os.path.join(
            DATA, "extracted_data_hlsp_files/csv/fov_6_degrees/binned_ir_data"
        ),
        "path_to_col_density_file": os.path.join(
            DATA, "column_density_along_sightlines.csv"
        ),
    }

    chi_square_grid_instance = ChiSquareGrid(star_id, config)
    A, G, chi_sq_grid = chi_square_grid_instance.compute_chi_square_grid()
    chi_square_grid_instance.plot_chi_square_grid(A, G, chi_sq_grid)
