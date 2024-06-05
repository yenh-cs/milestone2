"""
This module contains the code for the UTD data structure.

UTD contat
"""
import os
import h5py
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

    def filter_city(self, city):
        # city_chunks = []
        # for chunk in self.traffic_df:
        #     city_chunk = chunk.loc[chunk['city'] == city]
        #     city_chunks.append(city_chunk)
        #
        # if len(city_chunks) == 0:
        #     traffic_df = pd.DataFrame()
        # else:
        #     traffic_df = pd.concat(city_chunks)

        traffic_df = self.traffic_df.loc[self.traffic_df['city'] == city]

        detector_df = self.detector_df
        detector_df = detector_df.loc[detector_df['citycode'] == city]

        link_df = self.link_df
        link_df = link_df.loc[link_df['citycode'] == city]

        return self.utd_tuple(traffic_df, detector_df, link_df)
