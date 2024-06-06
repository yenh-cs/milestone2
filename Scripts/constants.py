"""
root: path to project's root
root_join: joins root with subpath
traffic_p: path to traffic df
detectors_p: path to detectors df
links_p: path to links df
cities_p: path to cities json
cities_d: dictionary of cities
"""
from configparser import ConfigParser
from Scripts.utd import UTD

config = ConfigParser()
config.read('config.ini')
utd_p = config['UTD']['path']
utd = UTD(utd_p)
