from Sparse_linear import SparseLinear2
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataLoader import csvDataset_mem
import sys
import torch
from torch.optim import SGD
import numpy as np


def get_indices(cpg_file, gene_exp_file, meta_file):
    with open(cpg_file, 'r') as f, open(gene_exp_file, 'r') as g, open(meta_file, 'r') as meta:
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
    return indices


def variance(dataset, idxs):
    cnt = 0
    base = dataset[idxs[0]][1]
    cum = torch.zeros_like(base)
    cumsq = torch.zeros_like(base)
    for i in idxs[1:]:
        cnt += 1
        expr = dataset[i][1]
        cum += (expr - base)
        cumsq += (expr - base) ** 2
    return cumsq**2/cnt - (cum/cnt)**2


device = "cuda"
gene_file = sys.argv[1]
cpg_file = sys.argv[2]
meta_file = sys.argv[3]
BATCHSIZE = int(sys.argv[4])
EPOCH = int(sys.argv[5])
save_path = sys.argv[6].strip('/')
npz_path = sys.argv[7]

indices = get_indices(cpg_file, gene_file, meta_file)
model = SparseLinear2(17113, 1, indices).to(device)
optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
dataset = csvDataset_mem([cpg_file, gene_file], npz_path)
split = int(len(dataset) * 0.8)
train_idx = list(range(split))
test_idx = list(range(split, len(dataset)))
trainLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(train_idx))
testLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(test_idx))
Loss = lambda x, y, w: ((x - y)**2 / w).sum()/x.shape[1]
best_loss = np.inf
train_var = variance(dataset, train_idx).to(device)
test_var = variance(dataset, test_idx).to(device)

for i in range(EPOCH):
    for cpg, gene_exp in trainLoader:
        cpg = cpg.to(device)
        gene_exp = gene_exp.to(device)
        optimizer.zero_grad()
        Loss(model(cpg).squeeze(2), gene_exp, train_var).backward()
        optimizer.step()
    for cpg, gene_exp in testLoader:
        running_loss = 0
        with torch.no_grad():
            cpg = cpg.to(device)
            gene_exp = gene_exp.to(device)
            running_loss += Loss(model(cpg).squeeze(2), gene_exp, test_var)
    test_loss = running_loss/len(test_idx)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model, '{}/best_model.pth'.format(save_path))
    print("epoch {} testing r2: {}".format(i+1, 1 - running_loss/len(test_idx)))
