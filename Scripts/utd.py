"""
This module contains the code for the UTD data structure.

UTD contat
"""
import json
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
        self._cities = None

    @property
    def cities(self):
        if self._cities is None:
            cities = os.listdir(self.root_dir)
            # rm hidden folders
            self._cities = [city for city in cities if city[0] != "."]
        return self._cities

    def _get_city_dir(self, city: str):
        return os.path.join(self.root_dir, city)

    def get_city_dfs(self, city: str):
        city_root = self._get_city_dir(city)
        df_traffic = pd.read_csv(os.path.join(city_root, self.traffic_filename))
        df_detector = pd.read_csv(os.path.join(city_root, self.detector_filename))
        df_link = pd.read_csv(os.path.join(city_root, self.link_filename))
        return self.utd_tuple(df_traffic, df_detector, df_link)

    def get_city_metadata(self, city: str):
        city_root = self._get_city_dir(city)
        metadata_p = os.path.join(city_root, "metadata.json")
        if not os.path.isfile(metadata_p):
            metadata = {}
        else:
            with open(metadata_p) as f:
                metadata = json.load(f)

        return metadata
