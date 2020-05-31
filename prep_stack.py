# スタッキング用にデータを分割

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.decomposition import PCA

# 読み込み
df_all = pd.read_csv("./data/all_prep.csv")

# nontree系モデル用の処理
# ダミー変数の作成
categorical_features = ["Embarked", "ticket_initials", "title", "age_bin"]
for col in categorical_features:
    tmp = pd.get_dummies(df_all[col], drop_first=True)
    tmp.columns = [f"dummy_{col}_{str(i)}" for i in tmp.columns]
    df_all = pd.concat([df_all, tmp], axis=1)

# trainとtestに再分割
df_train = df_all[~df_all["Survived"].isnull()].reset_index(drop=True)
df_test = df_all[df_all["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

# umap用のデータ
cols_umap = [i for i in df_test.columns if not "dummy" in i]
X_train_umap = df_train[cols_umap].drop("PassengerId", axis=1).values
X_test_umap = df_test[cols_umap].drop("PassengerId", axis=1).values

# umapで次元削減した特徴を加える
# 学習はtrainのみで行う
trans_umap = umap.UMAP(random_state=1031).fit(X_train_umap)
embed_train = trans_umap.transform(X_train_umap)
df_train["umap_1"] = embed_train[:, 0]
df_train["umap_2"] = embed_train[:, 1]
embed_test = trans_umap.transform(X_test_umap)
df_test["umap_1"] = embed_test[:, 0]
df_test["umap_2"] = embed_test[:, 1]

# pca用のデータ
X_train_pca = df_train.drop(categorical_features + ["Survived", "umap_1", "umap_2"], axis=1).values
X_test_pca = df_test.drop(categorical_features + ["umap_1", "umap_2"], axis=1).values

# 正規化
mmsc = MinMaxScaler()
X_train_norm = mmsc.fit_transform(X_train_pca)
X_test_norm = mmsc.transform(X_test_pca)

# PCAで次元削減した特徴を加える
# 学習はtrainのみで行う
n_components = 7
trans_pca = PCA(n_components=n_components).fit(X_train_norm)
print(np.cumsum(trans_pca.explained_variance_ratio_))
colnames = [f"pca_{i}" for i in range(n_components)]
df_train_pca = pd.DataFrame(trans_pca.transform(X_train_norm), columns=colnames)
df_test_pca = pd.DataFrame(trans_pca.transform(X_test_norm), columns=colnames)
df_train = pd.concat([df_train, df_train_pca], axis=1)
df_test = pd.concat([df_test, df_test_pca], axis=1)

# trainとvalidに分割
train, valid = train_test_split(df_train, random_state=1031, stratify=df_train["Survived"])

# tree系モデル用とnontree系モデル用に出力を変える
cols_tree = [i for i in df_test.columns if not "pca" in i]
cols_tree = [i for i in cols_tree if not "dummy" in i]
cols_nontree = ["PassengerId"] + [i for i in df_test.columns if "pca" in i]
train_tree = train[cols_tree + ["Survived"]]
valid_tree = valid[cols_tree + ["Survived"]]
test_tree = df_test[cols_tree]
train_nontree = train[cols_nontree + ["Survived"]]
valid_nontree = valid[cols_nontree + ["Survived"]]
test_nontree = df_test[cols_nontree]

# 出力
train_tree.to_csv("./data/train_prep_tree_base.csv", index=False)
valid_tree.to_csv("./data/train_prep_tree_valid.csv", index=False)
test_tree.to_csv("./data/test_prep_tree.csv", index=False)
train_nontree.to_csv("./data/train_prep_nontree_base.csv", index=False)
valid_nontree.to_csv("./data/train_prep_nontree_valid.csv", index=False)
test_nontree.to_csv("./data/test_prep_nontree.csv", index=False)
