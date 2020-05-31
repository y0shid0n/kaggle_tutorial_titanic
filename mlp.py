import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib
import random
import mlp_net

seed = 1031
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 読み込み
df_train = pd.read_csv("./data/train_prep_nontree_base.csv")
df_test = pd.read_csv("./data/train_prep_nontree_valid.csv")

# trainとvalidに分割
X = df_train.drop(["PassengerId", "Survived"], axis=1)
y = df_train["Survived"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1031, stratify=df_train["Survived"])

X_test = df_test.drop(["PassengerId", "Survived"], axis=1)

# gpuを使えるときはgpuを使う
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# Tensorに変換
X_train = torch.Tensor(X_train.values).to(device)
y_train = torch.LongTensor(y_train.values).to(device)
X_valid = torch.Tensor(X_valid.values).to(device)
y_valid = torch.LongTensor(y_valid.values).to(device)

X_test = torch.Tensor(X_test.values).to(device)

# Datasetを作成
ds_train = TensorDataset(X_train, y_train)

# 異なる順番で64個ずつデータを返すDataLoaderを作成
loader = DataLoader(ds_train, batch_size=64, shuffle=True)

net = mlp_net.net()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

train_losses = []
valid_losses = []

for epoch in tqdm(range(20)):
    running_loss = 0.0

    # ネットワークを学習モードにする（dropoutやbatch normalizationを有効化）
    net.train()

    for i, (x, y) in enumerate(loader):
        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / i)

    # ネットワークを評価モードにする（dropoutやbatch normalizationを有効化）
    net.eval()

    # validデータの損失関数を計算
    y_pred = net(X_valid)
    valid_loss = loss_fn(y_pred, y_valid)
    valid_losses.append(valid_loss.item())


# パラメータをハッシュ化してファイル名に投げる
hs = hashlib.md5(str(net).encode()).hexdigest()

# 予測
set_seed(seed)
y_pred_proba = net(X_test).cpu().detach().numpy().copy()
y_pred = torch.max(net(X_test), 1)[1].cpu().detach().numpy().copy()

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba[:, 1], name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)

output_proba.to_csv(f"./output/base_pred_proba_mlp_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_mlp_{hs}.csv", index=False)

# モデルの出力
torch.save(net.state_dict(), f"./pkl/mlp_{hs}.net")
