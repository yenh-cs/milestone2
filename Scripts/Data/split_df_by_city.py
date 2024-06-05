import os
import pandas as pd
from tqdm import tqdm


def split_df_by_city(
        df_traffic_p: str,
        df_detector_p: str,
        df_link_p: str,
        save_dir: str
):
    """
    Splits up the three datasets by city and saved in save_dir
    Args:
        df_traffic_p: path to df_traffic
        df_detector_p: path to df_detector
        df_link_p: path to df_link
        save_dir: path to root where city dfs will be saved

    Returns:
        None
    """
    OVERWRITE = False

    def make_dir(path, overwrite: bool):
        if overwrite:
            os.makedirs(path, exist_ok=True)
            return True

        try:
            os.makedirs(path, exist_ok=False)
        except FileExistsError:
            while True:
                s = input(
                    "Files already exists, would you like to overwrite, this will apply for all cities (y/n): "
                ).lower()
                if s == 'y':
                    make_dir(path, overwrite=True)
                elif s == 'n':
                    raise FileExistsError(f"{path} exits, please delete, rename, or move")
                else:
                    print("valid input is 'y' or 'n'")
    save_dir = os.path.join(save_dir, "UTD")
    OVERWRITE = make_dir(save_dir, OVERWRITE)
    df_detector = pd.read_csv(df_detector_p)
    df_link = pd.read_csv(df_link_p)
    df_traffic = pd.read_csv(df_traffic_p)

    cities = df_traffic['city'].unique()
    for city in tqdm(cities):
        city_traffic = df_traffic.loc[df_traffic['city'] == city]
        city_detector = df_detector.loc[df_detector['citycode'] == city]
        city_link = df_link.loc[df_link['citycode'] == city]
        city_dir = os.path.join(save_dir, city)
        make_dir(city_dir, OVERWRITE)

        dfs = [city_traffic, city_detector, city_link]
        df_names = ['traffic.csv', 'detector.csv', 'link.csv']

        for df, df_name in zip(dfs, df_names):
            save_p = os.path.join(city_dir, df_name)
            df.to_csv(save_p, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split traffic dfs by city and saves to directory")

    parser.add_argument("root_dir", type=str, help="root directory where other directories are located")
    parser.add_argument(
        "--traffic_csv",
        type=str,
        help="sub-path from root to traffic csv",
        nargs='?',
        default="traffic.csv"
    )
    parser.add_argument(
        "--detector_csv",
        type=str,
        help="sub-path from root to detector csv",
        nargs='?',
        default="detectors.csv"
    )
    parser.add_argument(
        "--link_csv",
        type=str,
        help="sub-path from root to link csv",
        nargs='?',
        default="links.csv"
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    traffic_csv_p = os.path.join(root_dir, args.traffic_csv)
    detector_csv_p = os.path.join(root_dir, args.detector_csv)
    link_csv_p = os.path.join(root_dir, args.link_csv)

    split_df_by_city(
        df_traffic_p=traffic_csv_p,
        df_detector_p=detector_csv_p,
        df_link_p=link_csv_p,
        save_dir=root_dir
    )
