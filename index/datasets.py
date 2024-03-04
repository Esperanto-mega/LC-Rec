import numpy as np
import torch
import torch.utils.data as data
import pandas as pd


class EmbDataset(data.Dataset):

    def __init__(self, data_path, std_norm = False, mean = 0, std = 1):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        # self.embeddings = np.load(data_path)
        names = ['emb']
        usecols = [2]
        tsv_data = pd.read_csv(data_path, sep = '\t',usecols = usecols, names = names, quotechar=None, quoting=3)
        features = tsv_data['emb'].values.tolist()
        num_data = len(features)
        for i in range(num_data):
            features[i] = [float(s) for s in features[i].split(' ')]
        if std_norm:
            self.embeddings = (np.array(features) - mean) / std
        else:
            self.embeddings = np.array(features)
        assert self.embeddings.shape[0] == num_data
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
