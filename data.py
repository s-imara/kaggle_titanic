import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def change_male_female_to_bynary_value(data):
    sex_mapping = {'male': 1, 'female': 0}
    return data.applymap(lambda s: sex_mapping.get(s) if s in sex_mapping else s)

def fillnavalue(data):
    #return data.fillna(train_data.mean())  # заменяем средними значениями по колонке
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data[['SibSp', 'Parch']] = data[['SibSp', 'Parch']].fillna(value=0)
    #data['Fare'] = data['Fare'].fillna(data['Fare'].min())
    data['Cabin'] = data[['Cabin']].fillna(value=0)
    cabin_mapping = {'A': 1, 'B': 2, 'C': 3}
    data['top_cabin'] = data['Cabin'].astype(str).str[0]
    data['top_cabin'] = data.applymap(lambda s: cabin_mapping.get(s) if s in cabin_mapping else s)
    return data

sns.set(color_codes=True)
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_data = change_male_female_to_bynary_value(train_data)
test_data = change_male_female_to_bynary_value(test_data)

train_data = fillnavalue(train_data)
test_data = fillnavalue(test_data)



# df.dropna() - удалить все наны
train_data_x = train_data[['Pclass', 'Sex', 'Age', 'Parch',  'top_cabin']].copy()
train_data_y = train_data[['Survived']].copy()
test_data_x = test_data[['Pclass', 'Sex', 'Age', 'Parch', 'top_cabin']].copy()
full_test_data = test_data




#sns.barplot(x=train_data.Sex,y=train_data.Age)
#plt.show(sns)






