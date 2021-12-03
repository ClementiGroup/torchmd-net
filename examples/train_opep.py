#!/usr/bin/env python
# coding: utf-8

# In[1]:

from torchmdnet.datasets.trainer import simple_train_loop
from torchmdnet.models import create_model
from torch_geometric.data import DataLoader
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.optim import Adam
import logging
import pickle
from datetime import datetime

model_name ='my_model'
logging.basicConfig(level=logging.INFO, filename='{}.log'.format(model_name), filemode='a')

#np.random.seed(1875432)
#torch.manual_seed(1875432)
import os

train_data_dir = '/net/data02/nickc/oct_train_res_exclusion_oct_min_repul/'
train_force_glob = train_data_dir+"*delta_forces*"
train_coord_glob = train_data_dir+'*coords*'
train_embed_glob = train_data_dir+"*embeds*"

train_dataset = InMemoryDataset(train_coord_glob, train_force_glob, train_embed_glob)

test_data_dir = '/net/data02/nickc/oct_test_res_exclusion_oct_min_repul/'
test_force_glob = test_data_dir+"*delta_forces*"
test_coord_glob = test_data_dir+'*coords*'
test_embed_glob = test_data_dir+"*embeds*"

test_dataset = InMemoryDataset(test_coord_glob, test_force_glob, test_embed_glob)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

# Hyperparameters
num_interactions = 5
hidden_channels = 128
num_filters = 128
rbf_type = 'expnorm'
rbf_num = 64
low_cut = 0.0
high_cut = 20.0
embed_size = 25
derivative = True
trainable_rbf = False
activation = 'tanh'
neighbor_embedding = False
cfconv_aggr = 'add'
lr=0.0001
max_neighbors=1000

# In[7]:

device = torch.device('cuda',2)

args = {'model': 'graph-network',
        'num_filters': num_filters,
        'num_layers': num_interactions,
        'embedding_dimension': hidden_channels,
        'rbf_type': rbf_type,
        'num_rbf': rbf_num,
        'trainable_rbf': trainable_rbf,
        'activation': activation,
        'neighbor_embedding': neighbor_embedding,
        'cutoff_lower': low_cut,
        'cutoff_upper': high_cut,
        'max_z': embed_size,
        'max_num_neighbors': max_neighbors,
        'derivative': True,
        'reduce_op': 'add',
        'dipole': False,
        'atom_filter': -1,
        'prior_model': None,
        'output_model': 'Scalar',
        'aggr': cfconv_aggr
}

with open(model_name+"_args.pkl", "wb") as argfile:
    pickle.dump(args, argfile)

model = create_model(args)
logging.info(model.__str__())
assert model.representation_model.aggr == 'add'
for block in model.representation_model.interactions:
    assert block.conv.aggr == 'add'
#model.to(device)

print(model)
with open(model_name+"_state_dict_init.pkl", "wb") as modelfile:
    pickle.dump(model.state_dict(), modelfile)

from torch.nn.functional import mse_loss
loss = mse_loss
optimizer = Adam(model.parameters(), lr=lr)

simple_train_loop(model, optimizer, loss,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  num_epochs=30,
                  starting_epoch=0,
                  device=device,
                  model_name=model_name,
                  model_save_freq=None,
                  print_freq=1000)
