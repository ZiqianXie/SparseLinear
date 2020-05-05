import torch
import numpy as np


class GraphConv(torch.nn.Module):
    def __init__(self, nl_ind, nl_value, dims, nonlinearity=torch.nn.ReLU):
        super().__init__()
        self.nl_ind = torch.nn.Parameter(nl_ind , requires_grad = False)  # normalized laplacian indicies
        self.nl_value = torch.nn.Parameter(nl_value)  # normalized laplacian value
        self.dims = list(dims)
        self.W = torch.nn.ParameterList()
        for i in range(len(self.dims)-1):
            self.W.append(torch.nn.Parameter(torch.randn(self.dims[i], self.dims[i+1])/np.sqrt(2*self.dims[i]), requires_grad=True))
        self.nonlinearity = nonlinearity(True)

    def forward(self, x):
        # x is of shape (batch, num_node, indim)
        num_batch = x.shape[0]
        num_node = x.shape[1]
        out = x
        for i in range(len(self.W)):
            selected = torch.index_select(out, 1, self.nl_ind[:, 1])
            product = torch.einsum("x, bxn -> bxn", self.nl_value, selected)
            out = self.nonlinearity(torch.zeros(num_batch, num_node,
                                    self.dims[i]).to(x.device).scatter_add_(1, self.nl_ind[:, :1].
                                                                            unsqueeze(0).expand(num_batch, -1, self.dims[i]),
                                                                            product).matmul(self.W[i]))
        return out

# gc = GraphConv(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]),
#               torch.tensor([-1., 1., 1., -2., 1., 1., -1.]), (2, 5, 5))
# out = gc(torch.rand(10, 3, 2))