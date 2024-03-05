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
    parser.add_argument("--id_save_path", type=str, default="", help="output directory for id result")
    parser.add_argument("--collision_save_path", type=str, default="", help="output directory for collision analysis result")
    parser.add_argument("--infer_batchsize", type=int, default=64, help="data batchsize for infer")
    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    return parser.parse_args()

# dataset = "Games"
# ckpt_path = "/zhengbowen/rqvae_ckpt/xxxx"
# output_dir = f"/zhengbowen/data/{dataset}/"
# output_file = f"{dataset}.index.json"
# output_file = os.path.join(output_dir,output_file)

infer_args = parse_args()
print('infer_args:',infer_args)

device = torch.device(infer_args.device)
data = EmbDataset(infer_args.data_path)

ckpt = torch.load(infer_args.ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]
print('args:',args)

model = RQVAE(in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim, layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn, loss_type=args.loss_type,
              quant_loss_weight=args.quant_loss_weight, kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters, sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=64, shuffle=False, pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=True)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

id_results = {
    'text': [],
    'rq_id': []
}

raw_data = pd.read_csv(infer_args.data_path, sep = '\t',usecols = [0], names = ['text'], quotechar = None, quoting = 3)
id_results['text'] = raw_data['text'].values.tolist()
assert len(all_indices_str) == len(id_results['text'])

for indice in all_indices:
    str_indice = ''
    for i in indice:
        str_indice = str_indice + str(i) + ' '
    id_results['rq_id'].append(str_indice[:-1])

# Save index results
df = pd.DataFrame(id_results)
df.to_csv(infer_args.id_save_path, sep = '\t', index = False, header = 0)

# Save collision analysis results
id2text = {}
for i in range(len(all_indices_str)):
    if all_indices_str[i] in id2text:
        id2text[all_indices_str[i]] = id2text[all_indices_str[i]] + '; ' + id2text[all_indices_str[i]]
    else:
        id2text[all_indices_str[i]] = id_results['text'][i]
collision_rate = (len(all_indices_str) - len(id2text)) / len(all_indices_str)
print('collision_rate:', collision_rate)
# sort id2text
sort_result = sorted(id2text.items(), key = lambda t:len(t[1]), reverse = True)
id2text = dict(sort_result)
with open(infer_args.collision_save_path,'w',newline='') as f:
    writer = csv.writer(f)
    for row in id2text.items():
        writer.writerow(row)




















'''
all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)

        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
'''
