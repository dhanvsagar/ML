import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn import svm

train_data_file = "train.csv"

df = pd.read_csv(train_data_file, header=0)

target = ['Survived']
features = ['Age', 'Pclass','Sex', 'Fare']

train_x  = df[features]
train_Y =  df[target]

#Check for null values in each column

print(train_x['Sex'].isnull().sum())
print(train_x['Age'].isnull().sum())
print(train_x['Pclass'].isnull().sum())
print(train_x['Fare'].isnull().sum())

#Null values in train_x['Age'], so fill them with the median of available values

train_x['Age'] = train_x['Age'].fillna(train_x['Age'].median())
print("After filling : ", train_x['Age'].isnull().sum())

# Fill Male= 0 and Female = 1 for train_x['Sex'] using apply()
d = {'male':0, 'female':1}
train_x['Sex'] = train_x['Sex'].apply(lambda x:d[x])

train_feature, test_feature, train_label, test_label = tts(train_x,train_Y, test_size=0.2, random_state=42)

model = svm.LinearSVC()

model.fit(train_feature, train_label)
prediction = model.predict(test_feature)

print("Accuracy : ", model.score(test_feature,test_label))