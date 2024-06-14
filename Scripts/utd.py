"""
This module contains the code for the UTD data structure.

UTD contat
"""
import json
import os
import pandas as pd
from collections import namedtuple

class UTDIterator:
    def __init__(self, utd, traffic_flag, detector_flag, link_flag):
        self._utd = utd
        self._traffic_flag = traffic_flag
        self._detector_flag = detector_flag
        self._link_flag = link_flag

    def __next__(self):
        for city in self._utd.cities:
            return self._utd.get_city_dfs(
                city,
                traffic_flag=self._traffic_flag,
                detector_flag=self._detector_flag,
                link_flag=self._link_flag
            )


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

    def get_city_dfs(self, city: str, traffic_flag=True, detector_flag=True, link_flag=True):
        """

        Args:
            city: city to query
            traffic_flag: whether to return df_traffic, default True
            detector_flag: whether to return df_detector, default True
            link_flag: whether to return df_link, default True

        Returns:

        """
        city_root = self._get_city_dir(city)
        def get_df(path, flag):
            return pd.read_csv(path) if flag else None

        df_traffic = get_df(os.path.join(city_root, self.traffic_filename), traffic_flag)
        if traffic_flag:
            df_traffic['day'] = pd.to_datetime(df_traffic['day'], format="%Y-%m-%d")
        df_detector = get_df(os.path.join(city_root, self.detector_filename), detector_flag)
        df_link = get_df(os.path.join(city_root, self.link_filename), link_flag)
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

    def get_iterator(self, traffic_flag=True, detector_flag=True, link_flag=True):
        return UTDIterator(self, traffic_flag, detector_flag, link_flag)
