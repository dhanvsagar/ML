import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

train_data_file = "train.csv"
test_data_file  = "test.csv"

df  = pd.read_csv(train_data_file, header=0)
df2 = pd.read_csv(test_data_file, header=0)

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

model = RandomForestClassifier()

model.fit(train_feature, train_label)
prediction = model.predict(test_feature)

print("Accuracy : ", model.score(test_feature,test_label))



##  Test Data from test.csv

test_x = df2[features]
print(test_x['Sex'].isnull().sum())
print(test_x['Age'].isnull().sum())
print(test_x['Pclass'].isnull().sum())
print(test_x['Fare'].isnull().sum())


#Filling Null values for Age and Fare

test_x['Age'] = test_x['Age'].fillna(test_x['Age'].median())
test_x['Fare'] = test_x['Age'].fillna(test_x['Fare'].median())

print("After filling : ", test_x['Age'].isnull().sum())
print("After filling : ", test_x['Fare'].isnull().sum())


# Fill Male= 0 and Female = 1 for train_x['Sex'] using apply()
test_x['Sex'] = test_x['Sex'].apply(lambda x:d[x])

prediction_actual = model.predict(test_x)
print(prediction_actual)

PassengerId = df2['PassengerId']
print(type(PassengerId))

output = pd.DataFrame(data = PassengerId)
output['PassengerId'] = PassengerId
output['Survived']    = prediction_actual

print(len(prediction_actual))
print(len(PassengerId))

output.to_csv("predictions.csv", index=False)
#np.savetxt("predictions.csv", prediction_actual, delimiter=',')