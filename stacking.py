import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import pickle
from datetime import datetime
import torch
#from torch import nn
import mlp_net

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# hash values
hash_lr = "ee5db903966c86d64a1040cb5ab8801d"
hash_rf = "ca783ea6fd88a5247e4e24821cd23ecf"
hash_knn = "4f1ffb98696e1a77802a5e7c61f0cb09"
hash_lgbm = "f9a7f16ed061fbefb81b345d331d60b9"
hash_catb = "42de8481ab2b6892b64d9c8b9f931821"
hash_svc = "22f4d2dcae8656741c1e39fd12f2b1d6"
hash_mlp = "f9d6489a2e41890659c64f284976ba14"

# ベースモデル読み込み
with open(f"./pkl/lr_{hash_lr}.pkl", "rb") as f:
    clf_lr = pickle.load(f)
with open(f"./pkl/rf_{hash_rf}.pkl", "rb") as f:
    clf_rf = pickle.load(f)
with open(f"./pkl/knn_{hash_knn}.pkl", "rb") as f:
    clf_knn = pickle.load(f)
# with open(f"./pkl/lgbm_{hash_lgbm}.pkl", "rb") as f:
#     clf_lgbm = pickle.load(f)
with open(f"./pkl/catb_{hash_catb}.pkl", "rb") as f:
    clf_catb = pickle.load(f)
with open(f"./pkl/svc_{hash_svc}.pkl", "rb") as f:
    clf_svc = pickle.load(f)

cvbooster = lgb.CVBooster()
for i in range(5):
    booster_file = f"./pkl/lgbm_cv_{hash_lgbm}_{i}.txt"
    tmp = lgb.Booster(model_file=booster_file)
    cvbooster._append(booster=tmp)
with open(f"./pkl/lgbm_cv_{hash_lgbm}_best_iter.txt") as f:
    cvbooster.best_iteration = int(f.read())

net = mlp_net.net()
net.load_state_dict(torch.load(f"./pkl/mlp_{hash_mlp}.net"))

# データ読み込み
df_test_tree = pd.read_csv("./data/test_prep_tree.csv")
df_test_nontree = pd.read_csv("./data/test_prep_nontree.csv")

df_lr = pd.read_csv(f"./output/base_pred_proba_lr_{hash_lr}.csv")
df_rf = pd.read_csv(f"./output/base_pred_proba_rf_{hash_rf}.csv")
df_knn = pd.read_csv(f"./output/base_pred_proba_knn_{hash_knn}.csv")
df_lgbm = pd.read_csv(f"./output/base_pred_proba_lgbm_cv_{hash_lgbm}.csv")
df_catb = pd.read_csv(f"./output/base_pred_proba_catb_{hash_catb}.csv")
df_svc = pd.read_csv(f"./output/base_pred_proba_svc_{hash_svc}.csv")
df_mlp = pd.read_csv(f"./output/base_pred_proba_mlp_{hash_mlp}.csv")

df_all = df_lr.rename(columns={"pred": "lr"})\
    .merge(df_rf, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "rf"})\
    .merge(df_knn, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "knn"})\
    .merge(df_lgbm, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "lgbm"})\
    .merge(df_catb, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "catb"})\
    .merge(df_svc, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "svc"})\
    .merge(df_mlp, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "mlp"})

# stack base predicts for training meta model
X_valid = df_all.drop(["PassengerId", "Survived"], axis=1)
y_valid = df_all["Survived"]

# train meta model
meta_model = LogisticRegression(random_state=1031)
meta_model.fit(X_valid, y_valid)

# prediction
X_test_tree = df_test_tree.drop("PassengerId", axis=1)
X_test_nontree = df_test_nontree.drop("PassengerId", axis=1)

pred_lr = clf_lr.predict_proba(X_test_nontree)[:, 1]
pred_rf = clf_rf.predict_proba(X_test_tree)[:, 1]
pred_knn = clf_knn.predict_proba(X_test_nontree)[:, 1]
# pred_lgbm = clf_lgbm.predict(X_test_tree)  # return probability
# pred_lgbm = [0 if i < 0.5 else 1 for i in pred_lgbm]
pred_catb = clf_catb.predict_proba(X_test_tree)[:, 1]
pred_svc = clf_svc.predict_proba(X_test_nontree)[:, 1]
pred_mlp = net(torch.Tensor(X_test_nontree.values).to(device)).cpu().detach().numpy().copy()[:, 1]
lgbm_pred_proba_list = cvbooster.predict(X_test_tree, num_iteration=cvbooster.best_iteration)
pred_lgbm = np.array(lgbm_pred_proba_list).mean(axis=0)
# pred_lgbm = [0 if i < 0.5 else 1 for i in lgbm_pred_proba]

stacked_preds = np.column_stack((pred_lr, pred_rf, pred_knn, pred_lgbm, pred_catb, pred_svc, pred_mlp))
meta_valid_pred = meta_model.predict(stacked_preds)

# 出力
output = pd.concat([df_test_tree["PassengerId"], pd.Series(meta_valid_pred, name="Survived")], axis=1)
# intにしておく（floatになってると正しくscore判定されないっぽい）
output["Survived"] = output["Survived"].apply(lambda x: int(x))
now = datetime.now().strftime('%Y%m%d%H%M%S')

output.to_csv(f"./output/stacking_{now}.csv", index=False)

# 各モデルの結果も含めて出力
output_all = pd.concat([output, pd.DataFrame(stacked_preds, columns=["lr", "rf", "knn", "lgbm", "catb", "svc", "mlp"])], axis=1)
output_all.to_csv(f"./output/stacking_all_{now}.csv", index=False)
