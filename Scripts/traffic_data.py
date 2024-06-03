import h5py
import pandas as pd
from collections import namedtuple

class UTD:
    utd_tuple = namedtuple("TrafficData", ["traffic_df", "detector_df", "link_df"])
    def __init__(self, traffic_p: str, detector_p: str, link_p: str):
        self._traffic_p = traffic_p
        self._detector_p = detector_p
        self._link_p = link_p

        # self.traffic_df = pd.read_csv(self._traffic_p)
        self._detector_df = pd.read_csv(detector_p)
        self._link_df = pd.read_csv(link_p)

    @property
    def traffic_df(self):
        return pd.read_csv(self._traffic_p)

    @property
    def detector_df(self):
        return self._detector_df.copy()

    @property
    def link_df(self):
        return self._link_df.copy()

    def filter_city(self, city):
        # city_chunks = []
        # for chunk in self.traffic_df:
        #     city_chunk = chunk.loc[chunk['city'] == city]
        #     city_chunks.append(city_chunk)
        #
        # if len(city_chunks) == 0:
        #     traffic_df = pd.DataFrame()
        # else:
        #     traffic_df = pd.concat(city_chunks)

        traffic_df = self.traffic_df.loc[self.traffic_df['city'] == city]

        detector_df = self.detector_df
        detector_df = detector_df.loc[detector_df['citycode'] == city]

        link_df = self.link_df
        link_df = link_df.loc[link_df['citycode'] == city]

        return self.utd_tuple(traffic_df, detector_df, link_df)

class UTDH5(h5py.File):
    utd_tuple = namedtuple("TrafficData", ["traffic_df", "detector_df", "link_df"])

    def __init__(self, h5_p):
        self._p = h5_p
        super().__init__(h5_p)

    def get_city_dfs(self, city: str):
        city_data = self[city]

        out_dfs = {}
        for df_name in city_data.keys():
            rows = {}
            for col in city_data[df_name].keys():
                data = city_data[df_name][col]
                rows[col] = data

            out_dfs[df_name] = rows

        utd = self.utd_tuple(
            traffic_df=pd.DataFrame(out_dfs['traffic']),
            detector_df=pd.DataFrame(out_dfs['detector']),
            link_df=pd.DataFrame(out_dfs["link"])
        )

        return utd


if __name__ == "__main__":
    from Scripts.constants import root_join
    import h5py
    import numpy as np

    traffic = {
        'interval': np.int32,
        "flow": np.float32,
        "occ": np.float16,
        'error': np.int8,
        'speed': np.int16
    }

    detector = {
        "length": np.float16,
        "pos": np.float16,
        "lanes": np.int8,
        "linkid": np.int16,
        "long": np.float16,
        "lat": np.float16
    }

    link = {
        "long": np.float16,
        "lat": np.float16,
        "order": np.int8,
        "piece": np.int8,
        "linkid": np.int16,
        "group": np.float16
    }
    dtypes = {'traffic': traffic, "link": link, "detector": detector}
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
