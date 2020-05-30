import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegressionCV
from datetime import datetime
import hashlib
import pickle

# 読み込み
df_train = pd.read_csv("./data/train_prep_base.csv")
df_test = pd.read_csv("./data/train_prep_valid.csv")

X_train = df_train.drop(["PassengerId", "Survived"], axis=1)
y_train = df_train["Survived"]

X_test = df_test.drop(["PassengerId", "Survived"], axis=1)

# 特徴量をランダムフォレストの重要度が高いやつに絞る
#cols = ["Sex", "title", "umap_2", "Pclass", "Fare", "umap_1", "Age"]
cols = ["Sex", "title", "umap_2", "umap_1", "Age", "Embarked", "ticket_1"]
X_train = X_train[cols]
X_test = X_test[cols]

#clf = LogisticRegressionCV(cv=5, scoring="roc_auc", random_state=1031)

param_grid = {
    "C": np.logspace(-3, 3, 7),
    "penalty": ["l1", "l2"],  # l1 lasso l2 ridge
    # const
    "random_state": [1031]
}

clf = GridSearchCV(LogisticRegression(),
                   param_grid,
                   scoring = "neg_log_loss",  # metrics
                   cv = 10,
                   n_jobs = -1
                  )
clf.fit(X_train, y_train)

print("Best Model Parameter: ", clf.best_params_)

# パラメータをハッシュ化してファイル名に投げる
hs = hashlib.md5(str(clf.best_params_).encode()).hexdigest()

clf = clf.best_estimator_ # best estimator

y_pred_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba[:, 1], name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)

now = datetime.now().strftime('%Y%m%d%H%M%S')

output_proba.to_csv(f"./output/base_pred_proba_lr_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_lr_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/lr_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
