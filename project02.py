# compare algorithms
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
names = ["Dataline","Play","PlayerLinenumber","ActSceneLine","Player","PlayerLine"]
dataset = read_csv('./archive/Shakespeare_data.csv', names=names)

dataset = dataset.dropna()

array = dataset.values
X = array[:,1]
print(len(X))
Y = array[:,4]
print(len(Y))

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
X = label_encoder.fit_transform(X)
Y = label_encoder.fit_transform(Y)
print (X)
print (Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X.reshape(-1,1), Y.reshape(-1,1), test_size=0.2, random_state=1, shuffle=True)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
predictions = predictions.astype(int)
print(accuracy_score(Y_validation, predictions))

X = array[:,2]
print(len(X))
Y = array[:,4]
print(len(Y))

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
X = label_encoder.fit_transform(X)
Y = label_encoder.fit_transform(Y)
print (X)
print (Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X.reshape(-1,1), Y.reshape(-1,1), test_size=0.2, random_state=1, shuffle=True)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
predictions = predictions.astype(int)
print(accuracy_score(Y_validation, predictions))


X = array[:,3]
print(len(X))
Y = array[:,4]
print(len(Y))

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
X = label_encoder.fit_transform(X)
Y = label_encoder.fit_transform(Y)
print (X)
print (Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X.reshape(-1,1), Y.reshape(-1,1), test_size=0.2, random_state=1, shuffle=True)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
predictions = predictions.astype(int)
print(accuracy_score(Y_validation, predictions))

X = array[:,5]
print(len(X))
Y = array[:,4]
print(len(Y))

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
X = label_encoder.fit_transform(X)
Y = label_encoder.fit_transform(Y)
print (X)
print (Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X.reshape(-1,1), Y.reshape(-1,1), test_size=0.2, random_state=1, shuffle=True)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
predictions = predictions.astype(int)
print(accuracy_score(Y_validation, predictions))
