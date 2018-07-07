from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts


wine = datasets.load_wine()

features    = wine.data
labels      = wine.target

print("Number of entries : ",len(features))


train_feature, test_feature, train_label, test_label = tts(features,labels, test_size=0.2)

#model = tree.DecisionTreeClassifier()
model = RandomForestClassifier()

#train
#model.fit(train_feature,train_label)
model.fit(train_feature,train_label)

#predict
predictions = model.predict(test_feature)


score = 0
for i in range(len(predictions)):
    if predictions[i] == test_label[i]:
        score = score+1
print("Accuracy : ", (score/ len(predictions)) * 100)