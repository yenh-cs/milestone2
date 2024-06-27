from copy import copy, deepcopy
from typing import List, Tuple
from torch.utils.data import Dataset
from Scripts.constants import utd
import numpy as np

class UTDCityDataset(Dataset):
    shallow_copy_keys = ['city', 'seq_len', 'stride', 'df_traffic', 'predict_len']
    deepcopy_keys = ['detid_data']

    def __init__(
            self,
            city: str,
            seq_len: int,
            predict_len: int,
            stride: int = 1
    ):
        """
        Creates a dataset from utd data
        Args:
            utd: utd
            seq_len: length of
            stride:
            cities:
        """
        Dataset.__init__(self)
        self.city = city
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.stride = stride

        if city not in utd.cities:
            raise ValueError(f'{city} not in utd.cities')

        dfs = utd.get_city_dfs(city, True, False, False)
        self.df_traffic = dfs.traffic_df
        # self.df_traffic.flow = self.df_traffic.flow / self.df_traffic.flow.max()

        self.detid_data = self.df_traffic.value_counts('detid').reset_index()
        self.detid_data['start_idx'] = 0
        self.add_seq_len_cumsum()
        print(self.detid_data.head().to_string())

    def __len__(self):
        try:
            return self.detid_data['seq_cumsum'].iloc[-1]
        except IndexError:
            return 0
        # return self.detid_data['seq_cumsum'].iloc[-1]

    def __getitem__(self, item):
        detid_idx, seq_idx = self.get_detid_seq_idx(self.detid_data.seq_cumsum, item)
        if detid_idx is None:
            raise StopIteration
        detid_data = self.detid_data.iloc[detid_idx]
        detid = detid_data.detid
        start_idx = detid_data.start_idx
        seq_idx += start_idx
        # dfs = utd.get_city_dfs(self.city, True, False, False)
        # traffic_df = dfs.traffic_df
        traffic_df = self.df_traffic.loc[self.df_traffic['detid'] == detid][['interval', "flow"]]
        flow = traffic_df.flow
        # TODO: interval is cyclic, messed up training
        # flow = traffic_df.sort_values('interval')['flow']
        x = flow.iloc[seq_idx: seq_idx + self.seq_len].values.astype(np.float32)[:, None]
        mx = x.max()
        y = flow.iloc[seq_idx + self.seq_len: seq_idx + self.seq_len + self.predict_len].values.astype(np.float32)
        y = np.pad(y, (0, self.predict_len - len(y)), constant_values=np.nan)[:, None]
        return x / mx, y / mx

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        shallow_d = {k: getattr(self, k) for k in self.shallow_copy_keys}
        deep_d = {k: deepcopy(getattr(self, k)) for k in self.deepcopy_keys}
        new_obj.__dict__.update(shallow_d)
        new_obj.__dict__.update(deep_d)
        return new_obj

    def add_seq_len_cumsum(self):
        self.detid_data['seq_len'] = self.detid_data["count"].apply(lambda r: self.num_of_sequences(r, self.seq_len, self.stride))
        self.detid_data['seq_cumsum'] = self.detid_data['seq_len'].cumsum()
        self.detid_data = self.detid_data.loc[self.detid_data.seq_len > 0]

    @staticmethod
    def num_of_sequences(n_rows: int, seq_len: int, stride: int) -> int:
        """
        Calculate the number of sequences that can be created from n_rows given sequence length and stride
        Args:
            n_rows: number of distinct timesteps
            seq_len: length of sequence
            stride: stride

        Returns:
            number of sequences that can be make
        """
        return (n_rows - seq_len) // stride

    @staticmethod
    def get_detid_seq_idx(series, i):
        ser = (i - series) < 0
        det_idx = ser.idxmax()
        if det_idx == 0:
            if not np.all(ser.values):
                return None, None
            seq_idx = i
        else:
            seq_idx = i - series.iloc[det_idx - 1]
        return det_idx, seq_idx

def train_val_test_split(
        utd_dataset: UTDCityDataset,
        train_p: float,
        val_p: float,
        test_p: float
):
    train_dset, val_dset, test_dset = utd_dataset, copy(utd_dataset), copy(utd_dataset)

    counts = train_dset.detid_data['count'].values
    train_counts = (counts * train_p).astype(int)
    val_counts = (counts * val_p).astype(int)
    test_counts = (counts * test_p).astype(int)

    val_start_idxs = train_counts
    test_start_idxs = val_start_idxs + test_counts

    train_dset.detid_data['count'] = train_counts
    val_dset.detid_data['count'] = val_counts
    test_dset.detid_data['count'] = test_counts

    val_dset.detid_data['start_idx'] = val_start_idxs
    test_dset.detid_data['start_idx'] = test_start_idxs

    train_dset.add_seq_len_cumsum()
    test_dset.add_seq_len_cumsum()
    val_dset.add_seq_len_cumsum()

    return train_dset, val_dset, test_dset

if __name__ == "__main__":
    utd_dset = UTDCityDataset('paris', 100, 10)
    train_dset, val_dset, test_dset = train_val_test_split(utd_dset, 0.8, 0.1, 0.1)
    n = len(train_dset)
    vals = train_dset[10]
    print(vals)



