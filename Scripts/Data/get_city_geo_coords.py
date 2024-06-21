import os
import json
from tqdm import tqdm
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="city_coordinates", timeout=100000)


def get_city_coords(city_name):
    location = geolocator.geocode(city_name)
    if location is not None:
        return location.latitude, location.longitude

def save_city_coords_to_utd(utd_path):
    cities = os.listdir(utd_path)

    for city in tqdm(cities):
        if city[0] == ".":
            continue
        data_p = os.path.join(utd_path, city, "metadata.json")

        if not os.path.exists(data_p):
            data = {}
        else:
            with open(data_p) as f:
                data = json.load(f)

        if "latitude" in data and "longitude" in data:
            continue
        else:
            lat, lon = get_city_coords(city)

            with open(data_p, 'w') as f:
                data['latitude'] = lat
                data['longitude'] = lon
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Saves city coordinates to each cities' metadata in UTD")
    parser.add_argument("utd_path", type=str, help="path to UTD root")

    args = parser.parse_args()
    save_city_coords_to_utd(args.utd_path)
