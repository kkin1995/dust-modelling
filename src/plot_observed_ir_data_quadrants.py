import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord    
import astropy.units as u 

data_path = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/data/extracted_data/fov_6_degrees/with_angle/"

plot_dir = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/plots/"

star_list = ["88469", "88496", "88506", "88380", "88581", "88560", "88256", "88142", "88705", "88463"]
quadrant_list = ["north", "south", "west", "east"]

for star in star_list:
    print("Plotting for Star {}".format(star))
    for quadrant in quadrant_list:
        print("--Plotting for quadrant {}".format(quadrant))
        #df = pd.read_csv("{}_ir_with_angles_{}.csv".format(star, quadrant))
        df = pd.read_csv(data_path + "{}_{}.csv".format(star, quadrant))

        angle = df["Angle"].values
        ir_flux = df["IR100"].values
        gl = df["GLON"].values
        gb = df["GLAT"].values
        gcs = SkyCoord(gl, gb, unit = u.deg, frame = 'galactic')

        plt.scatter(angle, ir_flux, s = 0.5)
        plt.title("IR Flux | HIP " + star + " | " + quadrant)
        plt.xlabel("Angle (Degrees)")
        plt.ylabel("IR Flux")
        plt.savefig(plot_dir + "IR_{}_{}.png".format(star, quadrant))
        plt.clf()

        plt.subplot(111, projection = 'aitoff')    
        plt.grid(True)    
        plt.scatter(gcs.l.wrap_at('180d').radian, gcs.b.radian) 
        plt.title("IR Flux | HIP " + star + " | " + quadrant)
        plt.savefig(plot_dir + "coordinate_plots/" + "IR_{}_{}_coordinates.png".format(star, quadrant))
        plt.clf()
