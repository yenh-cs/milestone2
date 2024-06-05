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

from Scripts.utd import UTD

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

utd_p = "/Users/joshfisher/PycharmProjects/Milestone2/Data/UTD"
utd = UTD(utd_p)

cities_p = root_join("Data/cities.json")
with open(cities_p) as f:
    cities_d = json.load(f)
