import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else 'cpu'

# nn.Sequentialはnn.Moduleの層を積み重ねてネットワークを構築する際に使用する
def net():
    net = nn.Sequential(
        nn.Linear(7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
        nn.Softmax(dim=1)
    ).to(device)
    return net
