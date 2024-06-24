import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tqdm import tqdm

"""
    Combines up the three datasets by city and saved in save_dir
    Args:
        df_traffic_p: path to df_traffic
        df_detector_p: path to df_detector
        df_link_p: path to df_link
        save_dir: path to city folder where city data is already saved

    Returns:
        None
"""

def clean_data_by_city(df):
    # covert day field to date type
    df['day'] = pd.to_datetime(df['day'], errors='coerce')

    # # fill all NaNs to zero
    # df = df.fillna(0)

    return df

def save_combined_data_by_city_utd_city(utd_path):
    cities = os.listdir(utd_path)

    for city in tqdm(cities):
        if city[0] == ".":
            continue

        #city folder path within UTD
        city_path = os.path.join(utd_path, city)
        detector_path = os.path.join(city_path, "detector.csv")
        link_path = os.path.join(city_path, "link.csv")
        traffic_path = os.path.join(city_path, "traffic.csv")

        df_detector_city = pd.read_csv(detector_path)
        df_link_city = pd.read_csv(link_path)
        df_traffic_city = pd.read_csv(traffic_path)

        # Rename 'city' in df_traffic to ''citycode' to match with df_detectors
        df_traffic_city.rename(columns={'city': 'citycode'}, inplace=True)

        # Merging detector and traffic datasets on 'detid'
        # Using 'inner' strategy to keep only those traffic measurements that have corresponding detector
        # location information to ensure we have the necessary spatial context for each traffic record
        df_det_tra = pd.merge(df_detector_city, df_traffic_city, on=['detid', 'citycode'], how='inner')

        # Merging det_tra_df and link datasets on # Merging merged_df with df_link_city on common columns: linkid and citycode
        # Using 'left' strategy to retain all traffic data while adding spatial context where available
        df_combined_data_city = pd.merge(df_det_tra, df_link_city, on=['linkid', 'citycode'], how='left')

        df_combined_data_city = clean_data_by_city(df_combined_data_city)

        combined_data_p = os.path.join(city_path, "combined_data_{city}.csv".format(city=city))
        df_combined_data_city.to_csv(combined_data_p)


if __name__ == "__main__":
    from pathlib import Path

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to the root directory (milestone2 in this case)
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    utd_path = os.path.join(root_dir, "Data/UTD")
    save_combined_data_by_city_utd_city(utd_path)



