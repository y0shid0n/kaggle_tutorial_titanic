import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import hashlib
import pickle

# 読み込み
df_train = pd.read_csv("./data/train_prep_tree_base.csv")
df_test = pd.read_csv("./data/train_prep_tree_valid.csv")

X_train = df_train.drop(["PassengerId", "Survived"], axis=1)
y_train = df_train["Survived"]

X_test = df_test.drop(["PassengerId", "Survived"], axis=1)

# trainとvalidに分割
#X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# use a full grid over all parameters
param_grid = {
    "max_depth": [3, 4, 5, 6, 7, None],
    "max_features": [5, 10, 15],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 3, 5],
    # const
    "n_estimators":[1000],
    "bootstrap": [True],
    "criterion": ["gini"],
    "random_state": [1031]
}

# fix_params
# param_grid = {
#     "max_depth": [4],
#     "max_features": [10],
#     "min_samples_split": [2],
#     "min_samples_leaf": [1],
#     "n_estimators":[1000],
#     "bootstrap": [True],
#     "criterion": ["gini"],
#     "random_state": [1031]
# }

clf = GridSearchCV(estimator = RandomForestClassifier(),
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

# 変数重要度を出力
with open(f"./output/importance_rf_{hs}.csv", "w", newline="", encoding="utf-8") as f:
    f.write("feature,importance\n")
    for score, name in sorted(zip(clf.feature_importances_, X_train.columns), reverse=True):
        f.write(f"{name},{score}\n")
        print(f"{name}: {score}")

# テストデータを予測する
y_pred_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba[:, 1], name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)

now = datetime.now().strftime('%Y%m%d%H%M%S')

output_proba.to_csv(f"./output/base_pred_proba_rf_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_rf_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/rf_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
