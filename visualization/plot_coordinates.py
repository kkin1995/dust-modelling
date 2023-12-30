import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nuv", action="store_true")
parser.add_argument("--fuv", action="store_true")
args = parser.parse_args()

data_dir = "/Users/karankinariwala/Library/CloudStorage/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/observed-uv-data/data/extracted_data/fov_6_degrees/"
coord_plot_dir = "/Users/karankinariwala/Library/CloudStorage/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/m8-dust-modeling/data/processed/"

star_list = [88463, 88705, 88506, 88560, 88581, 88380, 88142, 88469, 88496, 88256]

for star in star_list:
    print("Reading Data From {}".format(star))
    filename = data_dir + "extracted_observed_" + str(star) + ".csv"
    df = pd.read_csv(filename)
    if args.nuv:
        print("-- Reading NUV Data")
        df.drop(df.loc[df["NUV"] == -9999].index, inplace=True)
        df.drop(df.loc[df["NUV_STD"] == 1000000].index, inplace=True)
    elif args.fuv:
        print("-- Reading FUV Data")
        df.drop(df.loc[df["FUV"] == -9999].index, inplace=True)
        df.drop(df.loc[df["FUV_STD"] == 1000000].index, inplace=True)
    glon = df["GLON"].values
    glat = df["GLAT"].values
    glon[glon > 180] -= 360
    fig, ax = plt.subplots()
    ax.scatter(glon, glat, s=0.05)
    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")
    ticks = ax.get_xticks()
    ticks[ticks < 0] += 360
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([int(tick) for tick in ticks])
    if args.nuv:
        ax.set_title("HIP " + str(star) + " NUV")
        plt.savefig(coord_plot_dir + str(star) + "_nuv_coordinates.jpg")
    elif args.fuv:
        ax.set_title("HIP " + str(star) + " FUV")
        plt.savefig(coord_plot_dir + str(star) + "_fuv_coordinates.jpg")
    plt.clf()
