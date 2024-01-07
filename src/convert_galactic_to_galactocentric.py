import astropy.units as u
import astropy.coordinates as coord
import pandas as pd


def convert_to_galactocentric(df):
    """
    Converts Galactic coordinates to Galactocentric coordinates.

    Args:
    ----
    df (pd.DataFrame): DataFrame containing stellar data with columns "hip_id", "Distance(pc)", "gaia_l", and "gaia_b".

    Returns:
    ----
    pd.DataFrame: DataFrame with Galactocentric coordinates x, y, and z added.
    """

    distance_pc = df["Distance(pc)"].values
    gl = df["gaia_l"].values
    gb = df["gaia_b"].values

    c = coord.SkyCoord(
        l=gl * u.deg, b=gb * u.deg, distance=distance_pc * u.pc, frame="galactic"
    )

    c_galactocentric = c.transform_to(coord.Galactocentric)

    df["x_galactocentric"] = c_galactocentric.x.value
    df["y_galactocentric"] = c_galactocentric.y.value
    df["z_galactocentric"] = c_galactocentric.z.value

    return df


if __name__ == "__main__":
    m8_data_path = "/Users/karankinariwala/Dropbox/KARAN/1-College/MSc/4th-Semester/Dissertation-Project/flux-from-castelli-kurucz-spectra/data/"

    df = pd.read_csv(
        m8_data_path + "m8_stellar_data_gaia_hipparcos_with_computed_distance.csv"
    )

    df_with_galactocentric = convert_to_galactocentric(df)

    # Save or print the new DataFrame as required
    print(df_with_galactocentric)
