import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import umap

# 読み込み
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
combine = [df_train, df_test]
combine_name = ["train", "test"]

for df, name in zip(combine, combine_name):
    # passenger idを一旦分離（後で戻す）
    # trainはsurviveも一旦分離
    if name == "train":
        df_tmp = df.drop(["PassengerId", "Survived"], axis=1)
        df_id = df[["PassengerId", "Survived"]]
    else:
        df_tmp = df.drop(["PassengerId"], axis=1)
        df_id = df[["PassengerId"]]
    
    # Cabinは落とす
    df_tmp = df_tmp.drop("Cabin", axis=1)
    
    # Embarkedは最頻値で埋める
    df_tmp["Embarked"] = df_tmp["Embarked"].mask(df_tmp["Embarked"].isnull(), df_tmp["Embarked"].mode()[0])
    
    # 同一チケットの人数
    df_tmp["same_ticket"] = df_tmp.groupby("Ticket")["Sex"].transform("count")
    
    # Ticketの頭1文字をとる
    # 少ないやつは1まとめにする
    df_tmp["ticket_1"] = df_tmp["Ticket"].str[:1]
    df_tmp["ticket_1"] = df_tmp["ticket_1"].apply(lambda x: "4" if re.match("[4-9]", x) else x)
    df_tmp["ticket_1"] = df_tmp["ticket_1"].apply(lambda x: "O" if re.match("F|L|W", x) else x)
    
    # 性別を数値に置き換え
    df_tmp["Sex"] = df_tmp["Sex"].replace("male", 1).replace("female", 0)
    
    # Embarkedを数値に置き換え
    df_tmp["Embarked"] = df_tmp["Embarked"].replace("C", 0).replace("Q", 1).replace("S", 2)
    
    # 同乗している家族の数
    df_tmp["family_size"] = df_tmp["SibSp"] + df_tmp["Parch"] + 1
    
    # 単身フラグ
    df_tmp["is_alone"] = 0
    df_tmp["is_alone"] = df_tmp["is_alone"].mask(df_tmp["family_size"] == 1, 1)
    
    # 単身フラグ2
    # same_ticketとfamily_sizeが両方1だったら単身
    df_tmp["is_alone_2"] = 0
    df_tmp["is_alone_2"] = df_tmp["is_alone_2"].mask((df_tmp["same_ticket"] == 1) & (df_tmp["family_size"] == 1), 1)
    
    # 敬称
    # 数が少ないものはまとめる
    df_tmp["title"] = df_tmp["Name"].str.extract("([A-Za-z]+)\.", expand=False)
    df_tmp['title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    
    # 年齢を敬称ごとの平均値で埋める
    df_tmp.loc[(df_tmp.Age.isnull()) & (df_tmp.title=='Mr'), 'Age'] = df_tmp.Age[df_tmp.title=="Mr"].mean()
    df_tmp.loc[(df_tmp.Age.isnull()) & (df_tmp.title=='Mrs'), 'Age'] = df_tmp.Age[df_tmp.title=="Mrs"].mean()
    df_tmp.loc[(df_tmp.Age.isnull()) & (df_tmp.title=='Master'), 'Age'] = df_tmp.Age[df_tmp.title=="Master"].mean()
    df_tmp.loc[(df_tmp.Age.isnull()) & (df_tmp.title=='Miss'), 'Age'] = df_tmp.Age[df_tmp.title=="Miss"].mean()
    df_tmp.loc[(df_tmp.Age.isnull()) & (df_tmp.title=='Other'), 'Age'] = df_tmp.Age[df_tmp.title=="Other"].mean()
    
    # lightgbmは数値かbooleanしかダメなのでlabel encodingが必要
    for column in ["ticket_1", "title"]:
        target_column = df_tmp[column]
        le = LabelEncoder()
        le.fit(target_column)
        label_encoded_column = le.transform(target_column)
        df_tmp[column] = pd.Series(label_encoded_column).astype('category')
    
    # 不要なカラムを削除
    df_tmp = df_tmp.drop(["Name", "Ticket"], axis=1)
    
    # testデータのFareに欠損があるので中央値で埋める
    if name == "test":
        df_tmp["Fare"] = df_tmp["Fare"].mask(df_tmp["Fare"].isnull(), df_tmp["Fare"].median())
        
    # umapで次元削減した特徴を加える
    embedding = umap.UMAP().fit_transform(df_tmp.values)
    df_tmp["umap_1"] = embedding[:, 0]
    df_tmp["umap_2"] = embedding[:, 1]

    # idを戻す
    df_tmp = pd.concat([df_id, df_tmp], axis=1)
    
    # 出力
    df_tmp.to_csv(f"./data/{name}_prep.csv", index=False)
