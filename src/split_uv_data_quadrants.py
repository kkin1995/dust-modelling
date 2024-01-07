import numpy as np
import pandas as pd

data_dir = "../data/extracted_data/fov_6_degrees/with_angle/"
star_list = [88463, 88705, 88506, 88560, 88581, 88380, 88142, 88469, 88496, 88256]

star_data = pd.read_csv("../data/m8-stellar-data/m8_stellar_data_gaia_hipparcos_with_computed_distance.csv")

#star = 88469
for star in star_list:
    print("Extracting data in each quadrant for {}".format(star))

    star_glon = star_data[star_data["hip_id"] == star]["gaia_l"].values[0]
    star_glat = star_data[star_data["hip_id"] == star]["gaia_b"].values[0]

    extracted_data = pd.read_csv(data_dir + str(star) + ".csv")

    extracted_data_north = extracted_data[extracted_data["GLAT"] >= star_glat]
    extracted_data_south = extracted_data[extracted_data["GLAT"] <= star_glat]
    extracted_data_west = extracted_data[extracted_data["GLON"] <= star_glon]
    extracted_data_east = extracted_data[extracted_data["GLON"] >= star_glon]

    extracted_data_first_cartesian_quadrant = extracted_data[(extracted_data["GLAT"] >= star_glat) & (extracted_data["GLON"] >= star_glon)]
    extracted_data_second_cartesian_quadrant = extracted_data[(extracted_data["GLAT"] >= star_glat) & (extracted_data["GLON"] <= star_glon)]
    extracted_data_third_cartesian_quadrant = extracted_data[(extracted_data["GLAT"] <= star_glat) & (extracted_data["GLON"] <= star_glon)]
    extracted_data_fourth_cartesian_quadrant = extracted_data[(extracted_data["GLAT"] <= star_glat) & (extracted_data["GLON"] >= star_glon)]

    extracted_data_north.to_csv(data_dir + str(star) + "_north.csv", index = False)
    extracted_data_south.to_csv(data_dir + str(star) + "_south.csv", index = False)
    extracted_data_west.to_csv(data_dir + str(star) + "_west.csv", index = False)
    extracted_data_east.to_csv(data_dir + str(star) + "_east.csv", index = False)

    extracted_data_first_cartesian_quadrant.to_csv(data_dir + str(star) + "_first_cartesian_quadrant.csv", index = False)
    extracted_data_second_cartesian_quadrant.to_csv(data_dir + str(star) + "_second_cartesian_quadrant.csv", index = False)
    extracted_data_third_cartesian_quadrant.to_csv(data_dir + str(star) + "_third_cartesian_quadrant.csv", index = False)
    extracted_data_fourth_cartesian_quadrant.to_csv(data_dir + str(star) + "_fourth_cartesian_quadrant.csv", index = False)

#from astropy.coordinates import SkyCoord
#import astropy.units as u
#import matplotlib.pyplot as plt
#gcs = SkyCoord(extracted_data_east["GLON"], extracted_data_east["GLAT"], unit = u.deg, frame = 'galactic')    
    
#plt.subplot(111, projection = 'aitoff')    
#plt.grid(True)    
#plt.scatter(gcs.l.wrap_at('180d').radian, gcs.b.radian)    
    
#plt.show()
