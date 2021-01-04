import numpy as np
import torch
from torch import nn


class LSTM_Config:
    def __init__(self, feature_size, hidden_size=128, tower_h1=64, tower_h2=32, output_size=4, time_steps=150,
                 n_tasks=7):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.output_size = output_size
        self.tower_h1 = tower_h1
        self.tower_h2 = tower_h2
        self.n_tasks = n_tasks


class LSTM_Model(nn.Module):
    def __init__(self, config: LSTM_Config):
        super().__init__()
        self.config = config
        self.weights = torch.nn.Parameter(torch.from_numpy(np.ones(config.n_tasks).astype('float')).to(torch.float))

        self.lstm_1 = nn.LSTM(input_size=config.feature_size, hidden_size=config.hidden_size,
                              num_layers=config.time_steps, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=config.tower_h1),
            nn.ReLU(),
            nn.Dropout())
        self.tower_1 = nn.Sequential(
            nn.Linear(in_features=config.tower_h1, out_features=config.tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=config.tower_h2, out_features=1)
        )
        self.tower_2 = nn.Sequential(
            nn.Linear(in_features=config.tower_h1, out_features=config.tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=config.tower_h2, out_features=1)
        )
        self.tower_3 = nn.Sequential(
            nn.Linear(in_features=config.tower_h1, out_features=config.tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=config.tower_h2, out_features=1)
        )
        self.tower_4 = nn.Sequential(
            nn.Linear(in_features=config.tower_h1, out_features=config.tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=config.tower_h2, out_features=1)
        )

        nn.init.orthogonal_(self.fc[0].weight)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        # s, b, h = x.shape
        # x = x.reshape(-1, h)
        x = x[:, -1, :].reshape(x.shape[0], x.shape[2])
        x = self.fc(x)
        y_1 = self.tower_1(x)
        y_2 = self.tower_2(x)
        y_3 = self.tower_3(x)
        y_4 = self.tower_4(x)
        return y_1, y_2, y_3, y_4


class My_Train_Loss(nn.Module):
    def __init__(self):
        super(My_Train_Loss, self).__init__(self)
