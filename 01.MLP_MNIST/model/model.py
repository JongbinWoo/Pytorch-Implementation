from torch import nn, optim
from torch.nn import functional as F 

class MLP(nn.Module):
    def __init__(
        self,
        input_features=784,
        hidden_size=256,
        output_features=10
    ):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_features)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
    def __str__(self):
        num_parameters = 0
        for params in self.parameters():
            if params.requires_grad:
                num_parameters += len(params.reshape(-1))
        
        return super().__str__() + f'\n\nNumber of Trainable Parameters: {num_parameters:,d}\n'