import torch.optim as optim

def get_optimizer(optimizer_name='Adam', lr=0.001, params=None):
    optimizer =  optim.__dict__[optimizer_name]
    return optimizer(params, lr=lr)