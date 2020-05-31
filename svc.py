import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

param_grid = {
    "C": np.linspace(0.01, 100, 50),
    "gamma": np.linspace(0.01, 100, 50),
    # const
    "kernel": ["rbf"],
    "probability": [True],
    "random_state":[1031],
}

clf = GridSearchCV(estimator = SVC(),
                param_grid = param_grid,
                scoring="neg_log_loss",  # metrics
                cv = 10,  # cross-validation
                n_jobs = -1)  # number of core

clf.fit(X_train, y_train) # fit

print("Best Model Parameter: ", clf.best_params_)
print("Best score", clf.best_score_)

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

output_proba.to_csv(f"./output/base_pred_proba_svc_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_svc_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/svc_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
