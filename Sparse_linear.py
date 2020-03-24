import torch
import numpy as np
from collections import Counter


class SparseLinear(torch.nn.Module):
    def __init__(self, num_gene, dim_rep, row_ind):
        # ind is of shape (num_gene, num_nonzero)
        super().__init__()
        self.dim_x = num_gene
        self.dim_y = dim_rep
        self.ind = torch.tensor(row_ind).unsqueeze(0)
        self.nnz = self.ind.shape[-1]
        self.w = torch.nn.Parameter(torch.randn(self.dim_x, self.dim_y, self.nnz)/np.sqrt(self.nnz))

    def forward(self, x):
        # x is of shape (batch, num_cpg)
        gathered = x.unsqueeze(2).expand(-1, -1, self.nnz).gather(1, self.ind.expand(x.shape[0],-1, -1))
        return torch.einsum("gvc, bgc->bgv", self.w, gathered)


class SparseLinear2(torch.nn.Module):
    def __init__(self, num_gene, dim_rep, indices):
        # ind is a list of list or tuple of coordinate (i, j) that marks the nonzero entries in the weight
        super().__init__()
        self.ind = torch.tensor(sorted(indices))
        nnz_counter = Counter(map(lambda x: x[0], indices))
        median_nnz = np.median(np.asarray(list(nnz_counter.values())))
        self.num_gene = num_gene
        self.dim_x = len(indices)
        self.dim_y = dim_rep
        self.w = torch.nn.Parameter(torch.randn(self.dim_x, self.dim_y)/np.sqrt(median_nnz))

    def forward(self, x):
        # x is of shape (batch, num_cpg)
        num_batch = x.shape[0]
        selected = torch.index_select(x, 1, self.ind[:, 1])
        product = torch.einsum("xv, bx -> bxv", self.w, selected)
        return torch.zeros(num_batch, self.num_gene,
                           self.dim_y).scatter_add_(1, self.ind[:, :1].unsqueeze(0).expand(num_batch, -1, self.dim_y),
                                                    product)


# test
# s = SparseLinear(3, 5, 2, [[0],[1],[2]])
# s(torch.randn(10, 5)).sum().backward()
# s = SparseLinear2(3, 2, [[0, 0],[1, 1],[2, 1],[2, 2]])
# s(torch.randn(10, 5)).sum().backward()
# print(s.w.grad)
