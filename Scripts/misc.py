from geopy.geocoders import Nominatim
import numpy as np

def get_city_coords(city_name):
    geolocator = Nominatim(user_agent="city_coordinates")
    location = geolocator.geocode(city_name)
    if location is not None:
        return location.latitude, location.longitude


def mape(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    mape = (y_true.squeeze() - y_pred.squeeze()) / (y_true.squeeze() + 1e-7)
    mape = np.mean(mape)
    return mape

def mse(y_true, y_pred):
    """ Mean Square Error """
    return np.mean((y_true.squeeze() - y_pred.squeeze()) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def me(y_true, y_pred):
    return np.mean(y_true - y_pred)

if __name__ == "__main__":
    y_true = np.array([0.1, 0.5, 0.9])
    y_pred = np.array([0.1, 0.5, 0.8])
    print(mape(y_true, y_pred))

