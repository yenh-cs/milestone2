import numpy as np
import warnings
warnings.simplefilter(action='ignore')

from Scripts.constants import utd, cities_d, root_join
import h5py
import pandas as pd
from tqdm import tqdm

p = root_join("Data/utd2.h5")
f = h5py.File(p, 'w')

traffic = {
    'interval': "int32",
    "flow": "float32",
    "occ": "float16",
    'error': "int8",
    'speed': "int16"
}

detector = {
    "length": "float16",
    "pos": "float16",
    "lanes": "int8",
    "linkid": "int16",
    "long": "float16",
    "lat": "float16"
}

link = {
    "long": "float16",
    "lat": "float16",
    "order": "int8",
    "piece": "int8",
    "linkid": "int16",
    "group": "float16"
}
dtypes = {'traffic': traffic, "link": link, "detector": detector}


for city in cities_d:
    print(city)
    city_g = f.create_group(city)
    city_utd = utd.filter_city(city)
    dfs = [city_utd.detector_df, city_utd.link_df, city_utd.traffic_df]
    df_names = ['detector', 'link', 'traffic']
    for df, df_name in zip(dfs, df_names):
        print(f"\t{df_name}")
        g = city_g.create_group(df_name)
        for col in df:
            print(f'\t\t{col}')
            ser = df[col]
            if ser.dtype == object:
                try:
                    ser = ser.astype(int)
                except ValueError:
                    ser = ser.astype(str)
                    ser = ser.fillna('nan')

            data = ser.values
            g.create_dataset(col, data=data)

p = root_join("Data/utd2.h5")
utd = UTDH5(p)

paris_utd = utd.get_city_dfs('paris')
for df in paris_utd:
    print(df.head().to_string())

# f = h5py.File(root_join("Data/utd.h5"))
# out_f = h5py.File(root_join("Data/utd2.h5"), 'w')

# for city_name in f.keys():
#     print(city_name)
#     city_g = f[city_name]
#     out_city_g = out_f.create_group(city_name)
#     for df_name in city_g.keys():
#         print("\t", df_name)
#         df_g = city_g[df_name]
#         out_df_g = out_city_g.create_group(df_name)
#         for col_name in df_g.keys():
#             changed = False
#             data = df_g[col_name]
#             print("\t\t", f"{col_name}: {data.dtype}")
#             if col_name in dtypes[df_name]:
#                 changed = True
#                 data = data[:]
#                 dtype = dtypes[df_name][col_name]
#                 data = data.astype(dtype)
#
#             if changed:
#                 out_df_g.create_dataset(col_name, data=data, dtype=data.dtype, shape=data.shape, compression='lzf', chunks=True)
#             else:
#                 out_df_g.create_dataset(col_name, data=data, compression='lzf', chunks=True)



