# スタッキング用にデータを分割

import pandas as pd
from sklearn.model_selection import train_test_split

# 読み込み
df_train = pd.read_csv("./data/train_prep.csv")

# trainとvalidに分割
train, valid = train_test_split(df_train, random_state=1031)

# 出力
train.to_csv("./data/train_prep_base.csv", index=False)
valid.to_csv("./data/train_prep_valid.csv", index=False)
