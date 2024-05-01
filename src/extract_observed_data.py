import pandas as pd


def extract_and_preprocess_observed_data(
    df: pd.DataFrame, uv_or_ir: str
) -> pd.DataFrame:
    """
    Extracts UV or IR GALEX observations from DataFrame that was previously extracted from HLSP files
    using get_galex_hlsp_data.py.

    Parameters:
    ----
    df (pd.DataFrame): DataFrame from get_galex_hlsp_data.py.
    uv_or_ir (str): Accepts either 'uv' or 'ir'.
    """
    df = df[
        ["glon", "glat", "fuv_final", "nuv_final", "ir100", "ebv", "fuv_std", "nuv_std"]
    ]
    if uv_or_ir == "uv":
        df = df[df["fuv_final"] != -9999]
        df = df[df["nuv_final"] != -9999]
        df = df[df["fuv_std"] != 1000000]
        df = df[df["nuv_std"] != 1000000]
        df.drop(labels=["ir100"], axis="columns", inplace=True)
    elif uv_or_ir == "ir":
        df.drop(
            labels=["fuv_final", "nuv_final", "fuv_std", "nuv_std"],
            axis="columns",
            inplace=True,
        )
    return df


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    DATA = os.environ.get("DATA")
    hlsp_data_path = os.path.join(
        DATA, "extracted_data_hlsp_files", "hip_88469_fov_10.csv"
    )
    uv_or_ir = "uv"
    df = pd.read_csv(hlsp_data_path)
    df = extract_and_preprocess_observed_data(df, "uv")

    df.to_csv(
        os.path.join(
            DATA, "extracted_data_hlsp_files", f"hip_88469_fov_10_{uv_or_ir}.csv"
        ),
        index=False,
    )
