#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models
from sklearn.preprocessing import MinMaxScaler


#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='pipes_data', choices=['metr-la', 'pems-bay', 'pemsd7-m','pipes_data','pipes_data_v2','river_data','pems08' , 'Real_data_depth' , 'Real_data_inflow' , 'climate_2' , 'climate_3'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=3)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'rw_norm_adj', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 2')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    # print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    print('DEVICE: ')
    print(device) 
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    #blocks.append([1])
    blocks.append([args.n_pred])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    # dataset_path = '/home/skatakam1/GNN/STGCN/data'
    # dataset_path = os.path.join(dataset_path, args.dataset)
    # data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    
    dataset_path = '/home/wwang6/STGCN+MPNN/Real data/WW01_v3'
    file_name = 'WW01_v3.xlsx'
    
    file_path = os.path.join(dataset_path, file_name)
    
    # 使用 pandas 读取 Excel 文件，跳过前 3 行和前 1 列
    data = pd.read_excel(file_path, header=None)  # 不读取表头
    data = data.iloc[3:, 1:].values  # 从第 4 行、第 2 列开始读取数据部分
    
    # 获取数据行数
    data_col = data.shape[0]
    print("Loaded data shape:", data.shape)

    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * 0.1))
    len_test = int(math.floor(data_col * 0.2))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)


    file_path = '/home/wwang6/STGCN+MPNN/Real data/WW01_edge.csv'
    # @WW edge index
    data = pd.read_csv(file_path)

    data['Inlet Node'] = data['Inlet Node'].astype(str)
    data['Outlet Node'] = data['Outlet Node'].astype(str)

    edges_data = data[['Name', 'Inlet Node', 'Outlet Node']]
    edges_df = pd.DataFrame(edges_data, columns=["Name", "Inlet Node", "Outlet Node"])

    # 将 'OF-1' 替换为唯一索引，构造 edge_index，去除重复项
    # unique_nodes = pd.concat([edges_df["Inlet Node"], edges_df["Outlet Node"]], axis=0).drop_duplicates()
    unique_nodes = pd.concat([edges_df["Inlet Node"], edges_df["Outlet Node"]]).unique()
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

    edges_df["Inlet Node"] = edges_df["Inlet Node"].map(node_mapping)
    edges_df["Outlet Node"] = edges_df["Outlet Node"].map(node_mapping)

    edge_index = torch.tensor([edges_df["Inlet Node"].tolist(), edges_df["Outlet Node"].tolist()], dtype=torch.long).to(device)

    num_nodes = len(node_mapping)

    print("Edge data:\n", edges_data)
    print("Edge Index:\n", edge_index)
    print("Shape of Edge Index:", edge_index.shape)
    print("Node Mapping (Original Node IDs -> Index):")
    for original_node, index in node_mapping.items():
        print(f"{original_node} -> {index}")

    # @WW edge attr
    # Normalize edge features
    # Remove 0,1, and string columns
    for col in data.columns:
        if data[col].nunique() == 1 and (0 in data[col].unique() or 1 in data[col].unique()):
            data.drop(col, axis=1, inplace=True)
        elif data[col].dtype == object and data[col].str.isalpha().all():
            data.drop(col, axis=1, inplace=True)

    data = data.dropna(axis=1, how='all')
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    feature_columns = data.columns.drop(['Name', 'Inlet Node', 'Outlet Node','Time Max. Flow (M/D/Y)'])
    edge_features = data[feature_columns]

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(edge_features)
    #normalized_features = pd.DataFrame(normalized_features, columns=feature_columns)
    print('Edge attr shape: ')
    print(normalized_features.shape) 
    edge_attr = torch.tensor(normalized_features, dtype=torch.float32).to(device)
       

    return n_vertex, zscore, train_iter, val_iter, test_iter, edge_index, edge_attr

def prepare_model(args, blocks, n_vertex):
    #loss = nn.MSELoss()
    loss = nn.L1Loss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, edge_index, edge_attr):
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            
            y_pred = model(x, edge_index, edge_attr)#.view(len(x), -1)  # [batch_size, num_nodes]
            #y = y.view(len(x), -1)
            y = torch.squeeze(y)
            y_pred = torch.squeeze(y_pred)
            #print('ypred size')
            #print(y_pred.size())
            #print('y size')
            #print(y.size())
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter, edge_index, edge_attr)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
           format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n /12, val_loss, gpu_mem_alloc))

        if es.step(val_loss):
            # print('Early stopping.')
            break

@torch.no_grad()
def val(model, val_iter, edge_index, edge_attr):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x, edge_index, edge_attr)#.view(len(x), -1)
        y = torch.squeeze(y)
        y_pred = torch.squeeze(y_pred)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n/12)

@torch.no_grad() 
def test(zscore, loss, model, test_iter,edge_index, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter, edge_index, edge_attr)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore, edge_index, edge_attr)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    logger = logging.getLogger('stgcn+mpnn')
    logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter, edge_index, edge_attr = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter, edge_index, edge_attr)
    test(zscore, loss, model, test_iter,edge_index, args)
