import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = '/home/wwang6/STGCN+MPNN/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    elif dataset_name == 'pipes_data':
        n_vertex = 23
    elif dataset_name == 'pipes_data_v2':
        n_vertex = 23
    elif dataset_name == 'river_data':
        n_vertex = 23
    elif dataset_name == 'Real_data_depth':
        n_vertex = 23
    elif dataset_name == 'Real_data_inflow':
        n_vertex = 23
    elif dataset_name == 'pems08':
        n_vertex = 170
    elif dataset_name == 'climate_2' or dataset_name == 'climate_3':
        n_vertex = 100

    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = '/home/wwang6/STGCN+MPNN/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    
    
    print(train.shape)
    print(val.shape)
    print(test.shape)
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    #y = np.zeros([num, n_vertex])
    y = np.zeros([num, 1, n_pred, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        #y[i] = data[tail + n_pred - 1]
        y[i, :, :, :] = data[tail: tail + n_pred].reshape(1, n_pred, n_vertex)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
