from torch.utils.data import Dataset
import pickle as pkl
import numpy as  np
import torch
from os import path


def precompute_tell(files, save_path):
    p = []
    for file in files:
        f = open(file, 'r')
        pos = []
        while f.readline():
            pos.append(f.tell())
        f.close()
        p.append(pos[:-1])
    with open(save_path, 'wb') as f:
        pkl.dump(p, f)
    return p


class csvDataset(Dataset):
    def __init__(self, csv_files, pkl_pos=None):
        if pkl_pos is None:
            self.pos = precompute_tell(csv_files, "saved_pos.pkl")
        else:
            with open(pkl_pos, 'rb') as f:
                self.pos = pkl.load(f)
        self.fs = [open(csv_file, 'r') for csv_file in csv_files]

    def __del__(self):
        [f.close() for f in self.fs]

    def __len__(self):
        return len(self.pos[0])

    def __getitem__(self, idx):
        outs = []
        for i, f in enumerate(self.fs):
            f.seek(self.pos[i][idx])
            line = f.readline()
            outs.append(torch.from_numpy(np.fromstring(line[line.index(',')+1:-1], sep=',')).float())
        return outs
    

class csvDataset_mem(Dataset):
    def __init__(self, csv_files, npz_path):
        if not path.exists(npz_path):
            cpg, gene = csv_files
            self.cpg = np.empty((874, 395505))
            self.gene = np.empty((874, 17113))
            with open(cpg, 'r') as f, open(gene, 'r') as g:
                f.readline()
                g.readline()
                for i, (line1, line2) in enumerate(zip(f, g)):
                    self.cpg[i] = np.fromstring(line1[line1.index(',')+1:-1], sep=',')
                    self.gene[i] = np.fromstring(line2[line2.index(',')+1:-1], sep=',')
            np.savez(npz_path, cpg=self.cpg, gene=self.gene)
        else:
            data = np.load(npz_path)
            self.cpg = data['cpg']
            self.gene = data['gene']
        self.cpg = torch.from_numpy(self.cpg).float()
        self.gene = torch.from_numpy(self.gene).float()

    def __len__(self):
        return 874

    def __getitem__(self, idx):
        return [self.cpg[idx], self.gene[idx]]
                
                


