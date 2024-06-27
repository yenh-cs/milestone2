from geopy.geocoders import Nominatim
import numpy as np

def get_city_coords(city_name):
    geolocator = Nominatim(user_agent="city_coordinates")
    location = geolocator.geocode(city_name)
    if location is not None:
        return location.latitude, location.longitude


def mpe(y_true, y_pred):
    """ Mean Percentage Error """
    mape = (y_true.squeeze() - y_pred.squeeze()) / (y_true.squeeze() + 1e-7)
    mape = np.mean(mape)
    return mape

def mae(y_true, y_pred):
    """ Mean Absolute Error """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """ Root Mean Square Error """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mse(y_true, y_pred):
    """ Mean Square Error """
    return np.mean((y_true.squeeze() - y_pred.squeeze()) ** 2)


if __name__ == "__main__":
    y_true = np.array([0.1, 0.5, 0.9])
    y_pred = np.array([0.1, 0.5, 0.8])
    print(mpe(y_true, y_pred))

