"""
root: path to project's root
root_join: joins root with subpath
traffic_p: path to traffic df
detectors_p: path to detectors df
links_p: path to links df
cities_p: path to cities json
cities_d: dictionary of cities
"""
import os.path
from configparser import ConfigParser
from Scripts.utd import UTD

def _get_config(root):
    config_p = os.path.join(root, ".config.ini")

    if not os.path.isfile(config_p):
        print(f".config.ini not found at: {config_p}")
        raise FileNotFoundError(
            "Missing .config.ini in project root, please run python -m Scripts.setup_config.py from project root"
        )
    else:
        config = ConfigParser()
        config.read('/Users/joshfisher/PycharmProjects/Milestone2/.config.ini')
        utd_p = config['UTD']['path']
        utd = UTD(utd_p)
        return utd

root = os.path.abspath(os.path.join(__file__, "../.."))
utd = _get_config(root)
