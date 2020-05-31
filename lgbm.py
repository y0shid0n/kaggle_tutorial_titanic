import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from datetime import datetime
import hashlib
import pickle

# 読み込み
df_train = pd.read_csv("./data/train_prep_tree_base.csv")
df_test = pd.read_csv("./data/train_prep_tree_valid.csv")

X = df_train.drop(["PassengerId", "Survived"], axis=1)
y = df_train["Survived"]

X_test = df_test.drop(["PassengerId", "Survived"], axis=1)

# カテゴリカル変数
categorical_features = ["Pclass", "Sex", "is_alone_2", "title", 'Embarked', "ticket_initials", "cabin_initials"]

# trainとvalidに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1031)

# lightgbm用のオブジェクトに変換
trains = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
valids = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features, reference=trains)

lgbm_params = {
    'learning_rate': 0.01,
    'num_leaves': 30,
    'boosting_type' : 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    "verbose": 20,
}

# lgbm_params = {
#     'boosting': 'dart',          # dart (drop out trees) often performs better
#     'application': 'binary',     # Binary classification
#     'learning_rate': 0.01,       # Learning rate, controls size of a gradient descent step
#     'min_data_in_leaf': 20,      # Data set is quite small so reduce this a bit
#     'feature_fraction': 0.7,     # Proportion of features in each boost, controls overfitting
#     'num_leaves': 41,            # Controls size of tree since LGBM uses leaf wise splits
#     'metric': 'binary_logloss',  # Area under ROC curve as the evaulation metric
#     'drop_rate': 0.15
# }


# パラメータをハッシュ化してファイル名に投げる
hs = hashlib.md5(str(lgbm_params).encode()).hexdigest()

# 上記のパラメータでモデルを学習する
clf = lgb.train(lgbm_params, trains,
                  # モデルの評価用データを渡す
                  valid_sets=valids,
                  # 最大で 1000 ラウンドまで学習する
                  num_boost_round=1000,
                  # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
                  early_stopping_rounds=100)

# テストデータを予測する
y_pred_proba = clf.predict(X_test, num_iteration=clf.best_iteration)
y_pred = [0 if i < 0.5 else 1 for i in y_pred_proba]

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba, name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)
now = datetime.now().strftime('%Y%m%d%H%M%S')

output_proba.to_csv(f"./output/base_pred_proba_lgbm_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_lgbm_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/lgbm_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
