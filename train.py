from Sparse_linear import SparseLinear2
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataLoader import csvDataset
import sys
import torch
from torch.nn import MSELoss
from torch.optim import SGD


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


device = "cuda"
gene_file = sys.argv[1]
cpg_file = sys.argv[2]
meta_file = sys.argv[3]
BATCHSIZE = int(sys.argv[4])
EPOCH = int(sys.argv[5])
if len(sys.argv) > 6:
    pickled_pos = sys.argv[6]
else:
    pickled_pos = None

model = SparseLinear2(17113, 1, get_indices(gene_file, cpg_file, meta_file)).to(device)
optimizer = SGD(lr=1e-3, momentum=0.9)
dataset = csvDataset([cpg_file, gene_file], pickled_pos)
split = int(len(dataset) * 0.8)
train_idx = list(range(split))
test_idx = list(range(split, len(dataset)))
trainLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(train_idx), shuffle=True)
testLoader = DataLoader(dataset, batch_size=BATCHSIZE, sampler=SubsetRandomSampler(test_idx))
Loss = MSELoss()
for i in range(EPOCH):
    for cpg, gene_exp in trainLoader:
        cpg = torch.tensor(cpg).to(device)
        gene_exp = torch.tensor(gene_exp).to(device)
        optimizer.zero_grad()
        Loss(model(cpg).squeeze(2), gene_exp).backward()
        optimizer.step()
    for cpg, gene_exp in testLoader:
        running_loss = 0
        with torch.no_grad:
            cpg = torch.tensor(cpg).to(device)
            gene_exp = torch.tensor(gene_exp).to(device)
            running_loss += Loss(model(cpg).squeeze(2), gene_exp)
        print("epoch {} testing r2: {}".format(i+1, running_loss/len(test_idx)))
