import torch
import numpy as np


class GraphConv(torch.nn.Module):
    def __init__(self, nl_ind, nl_value, indim, outdim, nonlinearity=torch.nn.ReLU):
        super().__init__()
        self.nl_ind = nl_ind  # normalized laplacian indicies
        self.nl_value = nl_value  # normalized laplacian value
        self.out_dim = outdim
        self.W = torch.nn.Parameter(torch.randn(indim, outdim)/np.sqrt(2*indim))
        self.nonlinearity = nonlinearity(True)

    def forward(self, x):
        # x is of shape (batch, num_node, indim)
        num_batch = x.shape[0]
        num_node = x.shape[1]
        indim = x.shape[2]
        selected = torch.index_select(x, 1, self.nl_ind[:, 1])
        product = torch.einsum("x, bxn -> bxn", self.nl_value, selected)
        return self.nonlinearity(torch.zeros(num_batch, num_node,
                                             indim).scatter_add_(1, self.nl_ind[:, :1].\
                                                                        unsqueeze(0).expand(num_batch, -1, indim),
                                                                        product).matmul(self.W))

# gc = GraphConv(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]),
#               torch.tensor([-1., 1., 1., -2., 1., 1., -1.]), 2, 5)
# out = gc(torch.rand(10, 3, 2))