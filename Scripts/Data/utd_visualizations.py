import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import numpy as np
from tqdm import tqdm

from Scripts.constants import utd
from Scripts.cache_viz import cache_plot

@cache_plot
def plot_city_detectors(city: str, dark=False, links=False, overwrite=False):
    assert city in utd.cities, f"{city} not in UTD"
    if links:
        utd_city = utd.get_city_dfs(city)
    else:
        utd_city = utd.get_city_dfs(city, True, True, False)
    df_traff = utd_city.traffic_df
    df_detec = utd_city.detector_df
    df_link = utd_city.link_df

    df_traff = df_traff.groupby('detid')[['flow']].mean().reset_index()
    df = df_traff.merge(df_detec, on='detid', how='outer')
    df = df.dropna(subset=['lat', 'long', 'flow'])

    mapbox_style = "carto-darkmatter" if dark else "carto-positron"
    city_data = utd.get_city_metadata(city)
    lat, lon = city_data['latitude'], city_data['longitude']

    fig = px.scatter_mapbox(
        df, lat='lat', lon='long', mapbox_style=mapbox_style, color='flow', size='flow', zoom=12,
        center={'lat': lat, 'lon': lon}, hover_name='detid'
    )
    if links:
        add_trace = go.Scattermapbox(
            lat=df_link['lat'],
            lon=df_link['long'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                color='lightsalmon'
            ),
            customdata=df_link[['linkid', 'order']].values,
            hovertemplate="<b>linkid:</b> %{customdata[0]}<br>" +
                            "<b>order:</b> %{customdata[1]}<br>"
        )

        fig.add_trace(add_trace)
    return fig

@cache_plot
def plot_flow_distibutions(n_cols=3, overwrite=False):
    cities = utd.cities
    n_cities = len(cities)
    n_rows = n_cities // n_cols + n_cities % n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=True)

    for i, city in enumerate(tqdm(cities)):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        ax.title.set_text(city)

        utd_city = utd.get_city_dfs(city, True, False, False)
        traffic_df = utd_city.traffic_df

        sns.histplot(traffic_df, x='flow', ax=ax, log_scale=True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    res = plot_flow_distibutions(n_cols=4)
    plt.show()
    # fig = plot_city_detectors('paris', dark=False, links=True)
    # fig.show()
