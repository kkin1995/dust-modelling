import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from dotenv import load_dotenv

load_dotenv()
import os
from seaborn import set

set()

PLOTS = os.environ.get("PLOTS")

parser = argparse.ArgumentParser(description="Choose to read FUV / NUV data")
parser.add_argument("--fuv", action="store_true")
parser.add_argument("--nuv", action="store_true")
args = parser.parse_args()

if args.fuv:
    print("Reading FUV Data")
elif args.nuv:
    print("Reading NUV Data")
else:
    print("Incorrect Argument")

data_dir = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/data/extracted_data/fov_6_degrees/with_angle/"
plots_dir = PLOTS + "fov_6_degrees/"
star_list = [88463, 88705, 88506, 88560, 88581, 88380, 88142, 88469, 88496, 88256]
quadrants = ["north", "south", "east", "west"]
# quadrants = ["first_cartesian_quadrant", "second_cartesian_quadrant", "third_cartesian_quadrant", "fourth_cartesian_quadrant"]

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

for star in star_list:
    print("Reading data for star: {}".format(star))
    for quadrant in quadrants:
        print("-- Reading data for quadrant: {}".format(quadrant))
        files = data_dir + str(star) + "_" + quadrant + ".csv"
        df = pd.read_csv(files)

        if args.nuv:
            df.drop(df.loc[df["NUV"] == -9999].index, inplace=True)
            df.drop(df.loc[df["NUV_STD"] == 1000000].index, inplace=True)
            angle = df["Angle"]
            nuv = df["NUV"]
            plt.ylabel("NUV (Photon Units)")
            plt.xlabel("Angle (degrees)")
            # plt.ylim(0, 11000)
            plt.scatter(angle, nuv, s=0.01)
            plt.title("Observed NUV Background - " + str(star) + " - " + quadrant)
            plt.savefig(plots_dir + "NUV_" + str(star) + "_" + quadrant + ".jpg")
            plt.clf()
        elif args.fuv:
            df.drop(df.loc[df["FUV"] == -9999].index, inplace=True)
            df.drop(df.loc[df["FUV_STD"] == 1000000].index, inplace=True)
            fuv = df["FUV"]
            angle = df["Angle"]
            plt.xlabel("Angle (degrees)")
            plt.ylabel("NUV (Photon Units)")
            plt.scatter(angle, fuv, s=0.01)
            # plt.ylim(0, 10000)
            plt.title("Observed FUV Background - " + str(star) + " - " + quadrant)
            plt.savefig(plots_dir + "FUV_" + str(star) + "_" + quadrant + ".jpg")
            plt.clf()
