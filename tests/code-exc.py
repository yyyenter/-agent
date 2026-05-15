import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    # Sigmoid: 1 / (1 + e^(-x))
    return 1.0 / (1.0 + torch.exp(-x))


def relu(x : torch.Tensor):
    return torch.maximum(x, torch.tensor(0.0, device = x.device))


class swiglu(nn.modules):
    def __init__(self,dim,ffn_dim):
        self.w1 = nn.Linear(dim,ffn_dim)
        self.w2 = nn.Linear(dim,ffn_dim)
        self.w3 = nn.Linear(ffn_dim,dim,)
    def forward(self,x):
        h1 = self.w1(x)
        h2 = self.w2(x)
        out = h1*F.silu(h2)

class layernorm(nn.modules):
    def __init__(self,dim):
        self.Ex = 
    def forward(x):
        a = torch.sum(x)

x = torch.tensor([[-1.0, 0.0, 1.0],
                  [-2.0, 2.0, 3.0],
                  [-3.0, 3.0, 4.0]])


y = relu(x)
y1 = torch.sigmoid(x)
glue = nn.GELU()
y2 =glue(x)

if __name__ == "__main__":

    print(y2)
