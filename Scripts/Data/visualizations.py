import plotly.express as px
import pandas as pd

from Scripts.constants import utd

def plot_city_detectors(city: str, dark=False):
    utd_city = utd.get_city_dfs(city)
    df_traff = utd_city.traffic_df
    df_detec = utd_city.detector_df

    df_traff = df_traff.groupby('detid')[['flow']].mean().reset_index()
    df = df_traff.merge(df_detec, on='detid', how='outer')
    df = df.dropna(subset=['lat', 'long', 'flow'])

    mapbox_stype = "carto-darkmatter" if dark else "carto-positron"
    city_data = utd.get_city_metadata(city)
    lat, lon = city_data['latitude'], city_data['longitude']

    fig = px.scatter_mapbox(
        df, lat='lat', lon='long', mapbox_style=mapbox_stype, color='flow', size='flow', zoom=12,
        center={'lat': lat, 'lon': lon}
    )
    fig.show()

if __name__ == "__main__":
    plot_city_detectors('london')
