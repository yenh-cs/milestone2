"""
root: path to project's root
root_join: joins root with subpath
traffic_p: path to traffic df
detectors_p: path to detectors df
links_p: path to links df
cities_p: path to cities json
cities_d: dictionary of cities
"""

import os
import json

from Scripts.traffic_data import UTD

root = "/Users/joshfisher/PycharmProjects/Milestone2"

def root_join(subpath: str) -> str:
    """
    joins root path and subpath
    Args:
        subpath: path to be joined with root

    Returns:
        str: os.path.join(root, subpath)
    """
    return os.path.join(root, subpath)

traffic_p = root_join("Data/traffic.csv")
detectors_p = root_join("Data/detectors.csv")
links_p = root_join("Data/links.csv")
p = root_join("Data/utd2.h5")
# utd = UTDH5(p)
utd = UTD(traffic_p, detectors_p, links_p)

cities_p = root_join("Data/cities.json")
with open(cities_p) as f:
    cities_d = json.load(f)
