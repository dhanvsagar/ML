from sklearn import tree

# Height Weight shoe_size

x = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47], [175,64,39]]
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male']


clf = tree.DecisionTreeClassifier()
clf.fit(x,Y)

prediction = clf.predict([[190,70,43]])

print(prediction)