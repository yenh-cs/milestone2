from geopy.geocoders import Nominatim

def get_city_coords(city_name):
    geolocator = Nominatim(user_agent="city_coordinates")
    location = geolocator.geocode(city_name)
    if location is not None:
        return location.latitude, location.longitude


if __name__ == "__main__":
    import json

    p = "/Users/joshfisher/PycharmProjects/Milestone2/Data/cities.json"
    with open(p) as f:
        cities = json.load(f)

    out_d = {}
    for city in cities:
        lat_lon = get_city_coords(city)
        if lat_lon is not None:
            out_d[city] = {"latitude": lat_lon[0], "longitude": lat_lon[1]}
        else:
            out_d[city] = {'latitude': None, "longitude": None}

    with open(p, 'w') as f:
        json.dump(out_d, f, indent=4)
