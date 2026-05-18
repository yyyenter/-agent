import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

def sigmoid(x):
    # Sigmoid: 1 / (1 + e^(-x))
    return 1.0 / (1.0 + torch.exp(-x))


def relu(x : torch.Tensor):
    return torch.maximum(x, torch.tensor(0.0, device = x.device))


class swiglu(nn.Module):
    def __init__(self,dim,ffn_dim):
        self.w1 = nn.Linear(dim,ffn_dim)
        self.w2 = nn.Linear(dim,ffn_dim)
        self.w3 = nn.Linear(ffn_dim,dim,)
    def forward( self,x ):
        h1 = self.w1(x)
        h2 = self.w2(x)
        out = h1*F.silu(h2)

class layernorm(nn.Module):
    def __init__(self,dim, p=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))  # [dim]
        self.beta = nn.Parameter(torch.zeros(dim))  # [dim]
        self.p = p

    def forward(self, x):
        E_x = torch.mean( x, dim=-1, keepdim=True)
        D_x = ((x - E_x)**2).mean(dim = -1,keepdim =True)
        return self.gamma*(x-E_x)/torch.sqrt(D_x+self.p)+ self.beta

class RMS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim  # 默认在最后一维计算
    def forward(self, x):
        x_max = torch.max(x,dim=self.dim,keepdim=True)[0]
        exp_x = torch.exp(x-x_max)
        return exp_x/torch.sum(exp_x, dim=self.dim, keepdim=True)

class embedding(nn.Module):
    def __init__(self, dim, vocab_size,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.vocab_size = vocab_size
        self.Embedding = nn.Embedding(vocab_size, dim)
        self.out = nn.Linear(dim, vocab_size)
        self.out.weight = self.Embedding.weight
    def forward(self, x):
        return self.Embedding(x)

def kaiming(W,fan_in):
    std = torch.sqrt(2.0/fan_in)
    w = w.normal_(0,std)
    return w
    
class tokendrop(nn.Module):
    def __init__(self, drop_rate = 0.1):
        super().__init__()
        if drop_rate<0.0 or drop_rate>1.0:
            raise ValueError(f"Dropout概率必须在[0,1]之间，输入值：{drop_rate}")
        self.drop_rate = drop_rate
        self.scale = 1.0/(1 - drop_rate)
    def forward(self, x  ):
        if self.drop_rate == 0:
            return x
        keep_rate = 1.0 - self.drop_rate
        mask = torch.rand(x.shape[0], device=x.device) < keep_rate
        return x[mask]

B, L, D = 5, 3, 4
x = torch.randn(B, L, D).to(device)
target = torch.randint(0, D, (B, L)).to(device)
y = relu(x)
y1 = torch.sigmoid(x)
glue = nn.GELU()
y2 =glue(x)
layernorms = layernorm(D).to(device)
Softmaxs = Softmax()
ln = nn.LayerNorm(D).to(device)
y3 = ln(x)
y4 = Softmaxs(x)
embedding_layer = embedding(D, D).to(device)
y5 = embedding_layer(target)

if __name__ == "__main__":
    print(y5)
