from Sparse_linear import SparseLinear
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataLoader import csvDataset_mem
import sys
import torch
from torch.optim import Adam as optim
import numpy as np
from os import path
import pickle as pkl


def get_indices(cpg_file, gene_exp_file, meta_file, index_save_path):
    if path.exists(index_save_path):
        with open(index_save_path, 'rb') as f:
            return pkl.load(f)
    with open(cpg_file, 'r') as f, open(gene_exp_file, 'r') as g, open(meta_file, 'r') as meta, open(index_save_path, 'wb') as index_save_path:
        cpg_sites = list(map(lambda x: x[1:-1], f.readline().strip().split(',')[1:]))
        cpg_site_dict = dict(zip(cpg_sites, range(len(cpg_sites))))
        gene_exp = list(map(lambda x: x[1:-1], g.readline().strip().split(',')[1:]))
        gene_exp_dict = dict(zip(gene_exp, range(len(gene_exp))))
        meta.readline()
        indices = []
        for line in meta:
            gene, cpgs = line.strip().split(',')
            cpgs = cpgs.split(':')
            gene_idx = gene_exp_dict.get(gene, None)
            if gene_idx is None:
                continue
            for cpg in cpgs:
                cpg_idx = cpg_site_dict.get(cpg, None)
                if cpg_idx is None:
                    continue
                indices.append([gene_idx, cpg_idx])
        pkl.dump(indices, index_save_path)
    return indices


class running_mean_var:
    def __init__(self, eps=1e-8):
        self.sum = 0
        self.sumsq = 0
        self.cnt = 0
        self.eps = eps
 
    def update(self, x):
        with torch.no_grad():
            self.sum += x.sum(0)
            self.sumsq += (x**2).sum(0)
            self.cnt += x.shape[0]

    def get_mean_var(self):
        mean = self.sum/self.cnt
        return mean, (self.sumsq - mean*self.sum)/(self.cnt-1)+self.eps
        


def expr_mean_var(dataset, idxs, device):
    expr = dataset.gene[idxs]
    return expr.mean(0).to(device), expr.var(0).to(device)


device = "cuda"
gene_file = sys.argv[1]
cpg_file = sys.argv[2]
meta_file = sys.argv[3]
BATCHSIZE = int(sys.argv[4])
EPOCH = int(sys.argv[5])
save_path = sys.argv[6].strip('/')
npz_path = sys.argv[7]
index_save_path = sys.argv[8]
indices = get_indices(cpg_file, gene_file, meta_file, index_save_path)
model = SparseLinear(17113, 1, indices).to(device)
optimizer = optim(model.parameters(), lr=1e-3)
dataset = csvDataset_mem([cpg_file, gene_file], npz_path)
split = int(len(dataset) * 0.8)
train_idx = list(range(split))
test_idx = list(range(split, len(dataset)))
trainLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(train_idx))
testLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(test_idx))
Loss = lambda x, y, w: ((x - y)**2 / w).mean()
best_loss = np.inf
train_mean, train_var = expr_mean_var(dataset, train_idx, device)
test_mean, test_var = expr_mean_var(dataset, test_idx, device)
for i in range(EPOCH):
    out_estimator = running_mean_var()
    running_product = 0
    for cpg, gene_exp in trainLoader:
        cpg = cpg.to(device)
        gene_exp = gene_exp.to(device)
        optimizer.zero_grad()
        out = model(cpg).squeeze(2)
        out_estimator.update(out)
        with torch.no_grad():
            running_product += (out*gene_exp).sum(0)
        loss = Loss(out, gene_exp, train_var)
        loss.backward()
        optimizer.step()
    out_mean, out_var = out_estimator.get_mean_var()
    train_r2 = ((running_product/len(train_idx) - out_mean*train_mean)**2/out_var / train_var).mean()
    print("epoch {} training r2: {}".format(i+1, train_r2))
    running_product = 0
    out_estimator = running_mean_var()
    best_test_r2 = -np.inf
    for cpg, gene_exp in testLoader:
        with torch.no_grad():
            cpg = cpg.to(device)
            gene_exp = gene_exp.to(device)
            out = model(cpg).squeeze(2)
            out_estimator.update(out)
            running_product += (out*gene_exp).sum(0)
    out_mean, out_var = out_estimator.get_mean_var()
    test_r2 = ((running_product/len(test_idx) - out_mean*test_mean)**2/out_var / test_var).mean()
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        torch.save(model, '{}/best_model.pth'.format(save_path))
    print("epoch {} testing r2: {}".format(i+1, test_r2))