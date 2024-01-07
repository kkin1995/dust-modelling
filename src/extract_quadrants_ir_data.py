import pandas as pd

star_list = ["88469", "88496", "88506", "88380", "88581", "88560", "88256", "88142", "88705", "88463"] 

for star in star_list:
    print("Extracting Quadrant Data for Star {}".format(star))
    ir_data = pd.read_csv(star + "_ir_with_angles.csv")
    m8_data = pd.read_csv("/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/data/m8-stellar-data/m8_stellar_data_gaia_hipparcos_with_computed_distance.csv")

    star_gl = m8_data[m8_data["hip_id"] == int(star)]["gaia_l"].values[0]
    star_gb = m8_data[m8_data["hip_id"] == int(star)]["gaia_b"].values[0]

    ir_data_north = ir_data[ir_data["GB"] >= star_gb]
    ir_data_south = ir_data[ir_data["GB"] <= star_gb]
    ir_data_west = ir_data[ir_data["GL"] <= star_gl]
    ir_data_east = ir_data[ir_data["GL"] >= star_gl]

    ir_data_north.to_csv(star + "_ir_with_angles_north.csv", index = False)
    ir_data_south.to_csv(star + "_ir_with_angles_south.csv", index = False)
    ir_data_west.to_csv(star + "_ir_with_angles_west.csv", index = False)
    ir_data_east.to_csv(star + "_ir_with_angles_east.csv", index = False)
