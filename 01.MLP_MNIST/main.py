#%%
#data
from data_loader.data_loader import get_loader
from data_loader.dataset import get_mnist, get_augmentation


# %%
transform = get_augmentation()
train_dataset, test_dataset = get_mnist('./data', transform=transform)