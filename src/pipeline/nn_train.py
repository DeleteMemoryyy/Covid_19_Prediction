import warnings

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

warnings.filterwarnings('ignore')

import torch
import numpy as np
from dataloader import Dataloader
from DeepFM import DeepFM

validation_ratio = 0.2
batch_size = 500
lr = 0.00002
epochs = 10
verbose = True
print_every = 5
use_cuda = True

data_loader = Dataloader()
data_loader.load_from_csv()
data_loader.spilt_train_test()

x_train, x_valid, y_train, y_valid = train_test_split(data_loader.x_train, data_loader.y_train,
                                                      random_state=data_loader.rand_seed)
train_data = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
valid_data = TensorDataset(torch.from_numpy(x_valid.values), torch.from_numpy(y_valid.values))
test_data = TensorDataset(torch.from_numpy(data_loader.x_test.values), torch.from_numpy(data_loader.y_test.values))
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

data = data_loader.data
feature_sizes = []
for col in data.columns:
    if col in data_loader.continuous_column:
        feature_sizes.append(1)
    elif col == 'is_trade':
        continue
    else:
        feature_sizes.append(len(data[col].unique()) + 50)

net = DeepFM(feature_sizes=feature_sizes, continuous_field_size=len(data_loader.continuous_column), num_classes=2,
             use_cuda=use_cuda)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.03)

if use_cuda:
    net.cuda()
model = net.fit(train_loader, valid_loader, optimizer, epochs=epochs, verbose=verbose, print_every=print_every)

test_loss = net.validation(test_loader, model)
print('test_log_loss: {0:f}'.format(test_loss))
