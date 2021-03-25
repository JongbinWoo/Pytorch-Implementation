#%%
#data
from data_loader.data_loader import get_loader
from data_loader.dataset import get_mnist, get_augmentation

#config
from config import get_config

# %%
transform = get_augmentation()
train_dataset, test_dataset = get_mnist('./data', transform=transform)

train_loader = get_loader(train_dataset, batch_size=64, shuffle=True)
test_loader = get_loader(test_dataset, batch_size=64, shuffle=True)
# %%
config = get_config()
# %%
