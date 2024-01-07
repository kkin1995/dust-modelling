from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.sfd import SFDWebQuery


def get_col_density_from_dustmap(dustmap: str, gal_coord: tuple) -> float:
    """
    Uses the dustmaps package to download the E(B-V) along a line of sight given by
    galactic coordinates and converts the E(B-V) to column density (atoms cm-2) using the
    conversion factor from Bohlin, Savage, and Drake.

    Parameters:
    ----

    dustmap: str. Specifies the dustmap to use to query for the E(B-V). Currently, supports only "SFD".
    gal_coord: tuple. The coordinates in the galactic system for which to obtain the column density. (l, b)

    Returns:
    ----

    col_density: float. The column density of the dust along the line of sight given by gal_coord.
    """
    l0, b0 = gal_coord
    coords = SkyCoord(l0, b0, unit=u.deg, frame="galactic")
    if dustmap == "SFD":
        web_query = SFDWebQuery()

    ebv_map = web_query(coords)

    av = 2.742 * ebv_map
    rv = 3.1
    ebv = av / rv

    conversion_factor_col_density_ebv = 5.8e21  # From Bohlin, Savage and Drake
    col_density = ebv * conversion_factor_col_density_ebv  # atoms cm-2

    return col_density


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA = os.environ.get("DATA")
    gal_coord = (5.9717, -1.1760)

    col_density = get_col_density_from_dustmap("SFD", gal_coord)

    with open(os.path.join(DATA, "m8_col_density.yaml"), "w") as f:
        f.write(f"N(HI + H2): {col_density}")
