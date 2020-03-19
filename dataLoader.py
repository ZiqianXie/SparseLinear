from torch.utils.data import Dataset
import pickle as pkl
import numpy as  np


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
            self.pos = precompute_tell(csv_files, csv_files[0]+".pkl")
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
            outs.append(np.fromstring(line[line.index(',')+1:-1], sep=','))
        return outs


