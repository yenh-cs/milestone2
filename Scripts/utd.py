"""
This module contains the code for the UTD data structure.

UTD contat
"""
import os
import pandas as pd
from collections import namedtuple

class UTD:
    utd_tuple = namedtuple("TrafficData", ["traffic_df", "detector_df", "link_df"])
    traffic_filename = "traffic.csv"
    detector_filename = "detector.csv"
    link_filename = "link.csv"

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def get_city_dfs(self, city: str):
        city_root = os.path.join(self.root_dir, city)
        df_traffic = pd.read_csv(os.path.join(city_root, self.traffic_filename))
        df_detector = pd.read_csv(os.path.join(city_root, self.detector_filename))
        df_link = pd.read_csv(os.path.join(city_root, self.link_filename))
        return self.utd_tuple(df_traffic, df_detector, df_link)
