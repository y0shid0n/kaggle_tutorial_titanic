import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime

# ベースモデル読み込み
with open("./pkl/lr_33ccd66fcc615dbcbe55409d5829fb75.pkl", "rb") as f:
    clf_lr = pickle.load(f)
with open("./pkl/rf_183cebaf4b4c4179e0d8e4faf71f38fd.pkl", "rb") as f:
    clf_rf = pickle.load(f)
with open("./pkl/knn_891c09c3095237ef5e270a8efd5df9d3.pkl", "rb") as f:
    clf_knn = pickle.load(f)
with open("./pkl/lgbm_f9a7f16ed061fbefb81b345d331d60b9.pkl", "rb") as f:
    clf_lgbm = pickle.load(f)

# データ読み込み
df_test = pd.read_csv("./data/test_prep.csv")

df_lr = pd.read_csv("./output/base_pred_proba_lr_33ccd66fcc615dbcbe55409d5829fb75.csv")
df_rf = pd.read_csv("./output/base_pred_proba_rf_183cebaf4b4c4179e0d8e4faf71f38fd.csv")
df_knn = pd.read_csv("./output/base_pred_proba_knn_891c09c3095237ef5e270a8efd5df9d3.csv")
df_lgbm = pd.read_csv("./output/base_pred_proba_lgbm_f9a7f16ed061fbefb81b345d331d60b9.csv")

df_all = df_lr.rename(columns={"pred": "lr"})\
    .merge(df_rf, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "rf"})\
    .merge(df_knn, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "knn"})\
    .merge(df_lgbm, on=["PassengerId", "Survived"], how="inner")\
    .rename(columns={"pred": "lgbm"})


# stack base predicts for training meta model
X_valid = df_all.drop(["PassengerId", "Survived"], axis=1)
y_valid = df_all["Survived"]

# train meta model
meta_model = LogisticRegression()
meta_model.fit(X_valid, y_valid)

# prediction
X_test = df_test.drop("PassengerId", axis=1)

pred_lr = clf_lr.predict(X_test[["Sex", "title", "umap_2", "umap_1", "Age", "Embarked", "ticket_1"]])
pred_rf = clf_rf.predict(X_test)
pred_knn = clf_knn.predict(X_test[["Sex", "title", "umap_2", "umap_1", "Age", "Embarked", "ticket_1"]])
pred_lgbm = clf_lgbm.predict(X_test)

stacked_preds = np.column_stack((pred_lr, pred_rf, pred_knn, pred_lgbm))
meta_valid_pred = meta_model.predict(stacked_preds)

# 出力
output = pd.concat([df_test["PassengerId"], pd.Series(meta_valid_pred, name="Survived")], axis=1)
now = datetime.now().strftime('%Y%m%d%H%M%S')

output.to_csv(f"./output/stacking_{now}.csv", index=False)
