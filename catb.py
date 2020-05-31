import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import Pool
from catboost import cv
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
categorical_features = ["Pclass", "Sex", "is_alone_2", "title", 'Embarked', "ticket_initials", "cabin_initials", "pclass_sex"]

# trainとvalidに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1031, stratify=df_train["Survived"])

# catboost用のオブジェクトに変換
trains = Pool(X_train, label=y_train, cat_features=categorical_features)
valids = Pool(X_valid, label=y_valid, cat_features=categorical_features)

params = {
    'depth': 7,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'iterations': 1000,
    'verbose': 20,
    'random_seed': 1031
}

# 上記のパラメータでモデルを学習する
clf = CatBoostClassifier(**params)

# パラメータをハッシュ化してファイル名に投げる
hs = hashlib.md5(str(params).encode()).hexdigest()

clf.fit(trains, eval_set=valids, use_best_model=True, plot=True)

# feature importance
feature_importances = clf.get_feature_importance(trains)
feature_names = X_train.columns
with open(f"./output/importance_catb_{hs}.csv", "w", newline="", encoding="utf-8") as f:
    f.write("feature,importance\n")
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        f.write(f'{name},{score}\n')
        print(f'{name}: {score}')

# テストデータを予測する
y_pred_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# 出力
output_proba = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred_proba[:, 1], name="pred")], axis=1)
output = pd.concat([df_test[["PassengerId", "Survived"]], pd.Series(y_pred, name="pred")], axis=1)
now = datetime.now().strftime('%Y%m%d%H%M%S')

output_proba.to_csv(f"./output/base_pred_proba_catb_{hs}.csv", index=False)
output.to_csv(f"./output/base_pred_catb_{hs}.csv", index=False)

# モデルの出力
with open(f'./pkl/catb_{hs}.pkl', mode='wb') as fp:
    pickle.dump(clf, fp)
