#%%
#data
from data_loader.data_loader import get_loader
from data_loader.dataset import get_mnist, get_augmentation

#config
from config import get_config

#model
from model.model import MLP
from model import loss
from model import optimizer 

# %%
config = get_config()
# %%
transform = get_augmentation(**config.TRAIN.AUGMENTATION)
train_dataset, test_dataset = get_mnist(config.DATASET.ROOT, transform=transform)

train_loader = get_loader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
test_loader = get_loader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True)
# %%
mlp = MLP(input_features=784, hidden_size=256, output_features=config.DATASET.NUM_CLASSES)
optimizer = optimizer.get_optimizer(config.MODEL.OPTIM, config.TRAIN.BASE_LR, mlp.parameters())
import torch.nn as nn        ##
loss = nn.CrossEntropyLoss() ##
# %%
