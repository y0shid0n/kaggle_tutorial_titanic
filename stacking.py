import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime

# ベースモデル読み込み
with open("./pkl/lr_5263dece31afd0beb01a03b3e3a3731c.pkl", "rb") as f:
    clf_lr = pickle.load(f)
with open("./pkl/rf_3cf155877177efa6771973aa0c5b8059.pkl", "rb") as f:
    clf_rf = pickle.load(f)
with open("./pkl/knn_891c09c3095237ef5e270a8efd5df9d3.pkl", "rb") as f:
    clf_knn = pickle.load(f)
with open("./pkl/lgbm_f9a7f16ed061fbefb81b345d331d60b9.pkl", "rb") as f:
    clf_lgbm = pickle.load(f)
with open("./pkl/catb_42de8481ab2b6892b64d9c8b9f931821.pkl", "rb") as f:
    clf_catb = pickle.load(f)

# データ読み込み
df_test_tree = pd.read_csv("./data/test_prep_tree.csv")
df_test_nontree = pd.read_csv("./data/test_prep_nontree.csv")

df_lr = pd.read_csv("./output/base_pred_proba_lr_5263dece31afd0beb01a03b3e3a3731c.csv")
df_rf = pd.read_csv("./output/base_pred_proba_rf_3cf155877177efa6771973aa0c5b8059.csv")
df_knn = pd.read_csv("./output/base_pred_proba_knn_891c09c3095237ef5e270a8efd5df9d3.csv")
df_lgbm = pd.read_csv("./output/base_pred_proba_lgbm_f9a7f16ed061fbefb81b345d331d60b9.csv")
df_catb = pd.read_csv("./output/base_pred_proba_catb_42de8481ab2b6892b64d9c8b9f931821.csv")

df_all = df_lr.rename(columns={"pred": "lr"})\
    .merge(df_rf, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "rf"})\
    .merge(df_knn, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "knn"})\
    .merge(df_lgbm, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "lgbm"})\
    .merge(df_catb, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "catb"})

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
pred_lgbm = clf_lgbm.predict(X_test_tree)  # return probability
#pred_lgbm = [0 if i < 0.5 else 1 for i in pred_lgbm]
pred_catb = clf_catb.predict_proba(X_test_tree)[:, 1]

stacked_preds = np.column_stack((pred_lr, pred_rf, pred_knn, pred_lgbm, pred_catb))
meta_valid_pred = meta_model.predict(stacked_preds)

# 出力
output = pd.concat([df_test_tree["PassengerId"], pd.Series(meta_valid_pred, name="Survived")], axis=1)
# intにしておく（floatになってると正しくscore判定されないっぽい）
output["Survived"] = output["Survived"].apply(lambda x: int(x))
now = datetime.now().strftime('%Y%m%d%H%M%S')

output.to_csv(f"./output/stacking_{now}.csv", index=False)
