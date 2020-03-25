import torch
import numpy as np
from collections import Counter


class SparseLinear(torch.nn.Module):
    def __init__(self, num_gene, dim_rep, indices):
        # ind is a list of list or tuple of coordinate (i, j) that marks the nonzero entries in the weight
        super().__init__()
        self.ind = torch.nn.Parameter(torch.tensor(sorted(indices)).long(), requires_grad=False)
        nnz_counter = Counter(map(lambda x: x[0], indices))
        median_nnz = np.median(np.asarray(list(nnz_counter.values())))
        self.num_gene = num_gene
        self.dim_x = len(indices)
        self.dim_y = dim_rep
        self.w = torch.nn.Parameter(torch.randn(self.dim_x, self.dim_y).float()/np.sqrt(median_nnz))
        self.b = torch.nn.Parameter(torch.zeros(1, num_gene, dim_rep))

    def forward(self, x):
        # x is of shape (batch, num_cpg)
        num_batch = x.shape[0]
        selected = torch.index_select(x, 1, self.ind[:, 1])
        product = torch.einsum("xv, bx -> bxv", self.w, selected)
        return torch.zeros(num_batch, self.num_gene,
                           self.dim_y).to(x.device).scatter_add_(1, self.ind[:, :1].unsqueeze(0).expand(num_batch, -1, self.dim_y),
                                                    product) + self.b



# s = SparseLinear(3, 2, [[0, 0],[1, 1],[2, 1],[2, 2]])
# s(torch.randn(10, 5)).sum().backward()
# print(s.w.grad)
