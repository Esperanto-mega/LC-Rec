import collections
import json
import logging
import argparse

import numpy as np
import pandas as pd
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description = "Index")

    parser.add_argument("--data_path", type = str, default = "", help = "Infer data path.")
    parser.add_argument("--ckpt_path", type=str, default="", help="model checkpoint for infer")
    parser.add_argument("--save_path", type=str, default="", help="output directory for result")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    return parser.parse_args()

# Args for infer
infer_args = parse_args()
print(infer_args)

# Infer device
device = torch.device(infer_args.device)

# Load checkpoint
ckpt = torch.load(infer_args.ckpt_path, map_location = torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

data = EmbDataset(infer_args.data_path)

# Model checkpoint
model = RQVAE(in_dim = data.dim,
              num_emb_list = args.num_emb_list,
              e_dim = args.e_dim,
              layers = args.layers,
              dropout_prob = args.dropout_prob,
              bn = args.bn,
              loss_type = args.loss_type,
              quant_loss_weight = args.quant_loss_weight,
              kmeans_init = args.kmeans_init,
              kmeans_iters = args.kmeans_iters,
              sk_epsilons = args.sk_epsilons,
              sk_iters = args.sk_iters)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

# DataLoader
data_loader = DataLoader(data, num_workers = args.num_workers, batch_size = 64, shuffle = False, pin_memory = True)

# Infer
all_inference = []

for d in tqdm(data_loader):
    d = d.to(device)
    out = model(d)[0]
    # print(out.shape): torch.Size([64, 64])
    for i in range(out.shape[0]):
        all_inference.append(out[i].cpu().detach().numpy().tolist())

# Final results
all_results = {
    'text': [],
    'node_type': [],
    'rq_emb': []
}

names = ['text','node_type']
usecols = [0, 1]
raw_data = pd.read_csv(infer_args.data_path, sep = '\t',usecols = usecols, names = names, quotechar=None, quoting=3)
all_results['text'] = raw_data['text'].values.tolist()
all_results['node_type'] = raw_data['node_type'].values.tolist()

for emb in all_inference:
    str_emb = ''
    for x in emb:
        str_emb = str_emb + str(x) + ' '
    all_results['rq_emb'].append(str_emb[:-1])

# Write to tsv
df = pd.DataFrame(all_results)
# print(df)
# header = 0: w/o column name; index = False: w/o index column 
df.to_csv(infer_args.save_path, sep = '\t', header = 0, index = False)
