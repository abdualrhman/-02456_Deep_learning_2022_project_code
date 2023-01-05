import numpy as np
import pandas as pd
import pickle5 as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import torch
from torch.utils.data import Dataset

working_directory = os.getcwd()
file_path = working_directory + '/final_dataset.pickle'
with open(file_path, 'rb') as f:
    row_data = pickle.load(f)


class DataModel():
    def __init__(self, encoding_type='label', normalize_num=False, cols_to_ohe=[], cols_to_label_encode=[]):
        self.loaded_data = row_data.copy()
        cols_drop_list = ['voyage_id', 'service_id', 'voyage_no', 'voyage_scheduled_arrival_dt', 'voyage_scheduled_departure_dt', 'service_start_time_y', 'voyage_pol_id', 'voyage_pod_id', 'voyage_vessel_id',
                          'associate_crm_id', 'associate_id', 'customer_ref', 'service_type_id', 'unit_id', 'foreman_planned_place_on_board', 'service_start_time_x', 'service_end_time', 'prio_mapped', 'pob_std_mapped', 'pob_mapped']
        cols_to_norm = ['length', 'gross_weight', 'discharge_time_minutes']
        self.loaded_data.drop(cols_drop_list, axis=1, inplace=True)
        self.loaded_data.drop(self.loaded_data[
            (self.loaded_data.discharge_time_minutes == 0) |
            (self.loaded_data.discharge_time_minutes < 2) |
            (self.loaded_data.discharge_time_minutes > 350) |
            (self.loaded_data.gross_weight == 0)
        ].index,  inplace=True)
        self.loaded_data.dropna(inplace=True)
        self.df = pd.DataFrame()
        if encoding_type == 'one-hot':
            le_data = self.loaded_data.copy()
            if cols_to_label_encode:
                for col in cols_to_label_encode:
                    le = LabelEncoder()
                    le_data[col] = le.fit_transform(le_data[col])
            ohe_cols = ['unitype_id', 'place_on_board',
                        'parking_place', 'priority', 'deck_on_vessel', 'deck_stowed_order']
            ohe_cols = list(set(ohe_cols).difference(cols_to_label_encode))
            self.df = pd.get_dummies(le_data, columns=ohe_cols)
        elif encoding_type == 'label':
            cols_to_le = ['place_on_board', 'parking_place',
                          'priority', 'deck_stowed_order']
            le_cols = list(set(cols_to_le).difference(cols_to_ohe))
            le_data = self.loaded_data.copy()
            for col in le_cols:
                le = LabelEncoder()
                le_data[col] = le.fit_transform(le_data[col])
                assert len(le_data[col].unique()
                           ) == len(le_data[col].unique())
                # le_data.drop(le_cols, axis=1, inplace=True)
            assert le_data.shape[0] == self.loaded_data.shape[0]
            assert le_data.shape[1] == self.loaded_data.shape[1]
            if cols_to_ohe:
                le_data = pd.get_dummies(
                    le_data, columns=cols_to_ohe, dtype=float)
            self.df = le_data
        else:
            raise Exception("Unknowen encoding type: " + encoding_type)
        # if cols_to_ohe and encoding_type != 'one-hot':
        #     self.df = pd.get_dummies(
        #         self.loaded_data, columns=cols_to_ohe, dtype=float)
        #     # self.df_to_float()

        if normalize_num:
            self.df[cols_to_norm] = (self.df[cols_to_norm] - self.df[cols_to_norm].min()) / (
                self.df[cols_to_norm].max() - self.df[cols_to_norm].min())

        self.X = self.df.loc[:, self.df.columns != 'discharge_time_minutes']
        self.y = self.df['discharge_time_minutes']
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     self.X, self.y, random_state=0, train_size=.8)
        self.X_train, X_rem, self.y_train, y_rem = train_test_split(
            self.X, self.y, train_size=0.8, shuffle=True)
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
            X_rem, y_rem, test_size=0.5, shuffle=True)

    def get_df(self):
        return self.df

    # X, y
    def get_inputs_targets(self):
        return self.X, self.y

    # X_train, y_train
    def get_train_data(self):
        return self.X_train, self.y_train

    # X_valid, y_valid
    def get_valid_data(self):
        return self.X_valid, self.y_valid

    # X_test, y_test
    def get_test_data(self):
        return self.X_test, self.y_test


class TensorDataSet(Dataset):
    def __init__(self,  data_type='train', encoding_type='label', normalize_num=True,  cols_to_ohe=[], cols_to_label_encode=[], cols_to_drop=[]):
        dm = DataModel(encoding_type=encoding_type,
                       normalize_num=normalize_num, cols_to_ohe=cols_to_ohe, cols_to_label_encode=cols_to_label_encode)
        if data_type == 'train':
            X_data, y_data = dm.get_train_data()
        elif data_type == 'valid':
            X_data, y_data = dm.get_valid_data()
        elif data_type == 'test':
            X_data, y_data = dm.get_test_data()
        else:
            raise Exception("Unknowen data type: " + data_type)
        self.X = torch.tensor(X_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.float32)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TensorTabDataSet(Dataset):
    def __init__(self, data_type='train', normalize_num=True, encoding_type='label', cols_to_ohe=[], cols_to_label_encode=[]):
        dm = DataModel(encoding_type=encoding_type,
                       normalize_num=normalize_num, cols_to_ohe=cols_to_ohe, cols_to_label_encode=cols_to_label_encode)
        if data_type == 'train':
            X_data, y_data = dm.get_train_data()
        elif data_type == 'valid':
            X_data, y_data = dm.get_valid_data()
        elif data_type == 'test':
            X_data, y_data = dm.get_test_data()
        else:
            raise Exception("Unknowen data type: " + data_type)

        self.X = torch.tensor(X_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx, 2:].long(), self.X[idx, 0:2]), self.y[idx]
