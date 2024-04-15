import pandas as pd
import matplotlib.pyplot as plt
import argparse
from dotenv import load_dotenv
import os

load_dotenv()
DATA = os.environ.get("DATA")

parser = argparse.ArgumentParser()
parser.add_argument("--nuv", action="store_true")
parser.add_argument("--fuv", action="store_true")
parser.add_argument("--ir", action="store_true")
args = parser.parse_args()

data_dir = os.path.join(DATA, "extracted_data_hlsp_files/csv/fov_6_degrees")

plot_dir = os.path.join(DATA, "plots")

star_data = pd.read_csv(os.path.join(DATA, "m8_stellar_data_gaia_hipparcos.csv"))

for idx, row in star_data.iterrows():
    star = row["hip_id"]
    print("Reading Data From {}".format(star))
    filename = os.path.join(data_dir, "extracted_observed_" + str(star) + ".csv")
    df = pd.read_csv(filename)
    if args.nuv:
        print("-- Reading NUV Data")
        df.drop(df.loc[df["NUV"] == -9999].index, inplace=True)
        df.drop(df.loc[df["NUV_STD"] == 1000000].index, inplace=True)
    elif args.fuv:
        print("-- Reading FUV Data")
        df.drop(df.loc[df["FUV"] == -9999].index, inplace=True)
        df.drop(df.loc[df["FUV_STD"] == 1000000].index, inplace=True)
    elif args.ir:
        print("-- Reading IR Data")

    glon = df["GLON"].values
    glat = df["GLAT"].values
    glon[glon > 180] -= 360

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plt.axis("equal")
    plt.grid(True)
    ax.scatter(row["gaia_l"], row["gaia_b"], c="red", s=5, label=f"HIP {star} (star)")
    ax.scatter(glon, glat, s=0.05, label="Data")
    ax.set_xlabel("Galactic Longitude (deg)")
    ax.set_ylabel("Galactic Latitude (deg)")
    plt.legend()
    ticks = ax.get_xticks()
    ticks[ticks < 0] += 360
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([int(tick) for tick in ticks])
    if args.nuv:
        ax.set_title("HIP " + str(star) + " NUV")
        plt.savefig(os.path.join(plot_dir, str(star) + "_nuv_data_distribution.jpg"))
    elif args.fuv:
        ax.set_title("HIP " + str(star) + " FUV")
        plt.savefig(os.path.join(plot_dir, str(star) + "_fuv_data_distribution.jpg"))
    elif args.ir:
        ax.set_title("HIP " + str(star) + " IR100")
        plt.savefig(os.path.join(plot_dir, str(star) + "_ir_data_distribution.jpg"))
    plt.clf()
