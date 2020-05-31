import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import hashlib
import pickle

# 読み込み
df_train = pd.read_csv("./data/train_prep_nontree_base.csv")
df_test = pd.read_csv("./data/train_prep_nontree_valid.csv")

X_train = df_train.drop(["PassengerId", "Survived"], axis=1)
y_train = df_train["Survived"]

X_test = df_test.drop(["PassengerId", "Survived"], axis=1)

# umapで圧縮した次元のみに絞る
# cols = ["umap_1", "umap_2"]
# X_train = X_train[cols]
# X_test = X_test[cols]

# 特徴量をランダムフォレストの重要度が高いやつに絞る
# cols = ["Sex", "title", "umap_2", "Pclass", "Fare", "umap_1", "Age"]
# cols = ["Sex", "title", "umap_2", "umap_1", "Age", "Embarked", "ticket_initials"]
# X_train = X_train[cols]
# X_test = X_test[cols]

param_grid = {
    "n_neighbors": np.array(range(1, 10, 2)),  # 奇数
    "weights": ["uniform", "distance"],
    "p": [1, 2],
    # const
    "algorithm":["auto"],
}

clf = GridSearchCV(estimator = KNeighborsClassifier(),
                param_grid = param_grid,
                scoring="neg_log_loss",  # metrics
                cv = 10,  # cross-validation
                n_jobs = -1)  # number of core

clf.fit(X_train, y_train) # fit

print("Best Model Parameter: ", clf.best_params_)

# パラメータをハッシュ化してファイル名に投げる
hs = hashlib.md5(str(clf.best_params_).encode()).hexdigest()

clf = clf.best_estimator_ # best estimator

# テストデータを予測する
y_pred_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba[:, 1], name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)

now = datetime.now().strftime('%Y%m%d%H%M%S')

output_proba.to_csv(f"./output/base_pred_proba_knn_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_knn_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/knn_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
