# -*- coding: utf-8 -*-
"""


@author: sudet
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time

df = pd.read_csv('diabetes.csv', index_col=0)
#EDA
df.info()
df.isnull().sum()  #there is no missing data

# Seperate Features and Columns
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#Split into Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

#find best k value for knn 
acc = []
for i in range(1,40):
    neighors = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neighors.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=12, p=1)
#model training
knn.fit(X_train,y_train)
end_time = time.time()
print('\nKNN Clasification Train time: {}'.format(end_time - start_time))
start_time = time.time()
y_pred = knn.predict(X_test)
end_time = time.time()
print('\nKNN Clasification Predict time: {}'.format(end_time - start_time))
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print('\nKNN Confusion Matrix (without preprocessing):', cm)
print('\nKNN Accuracy Score (without preprocessing): ',acc) 

#DTC
from sklearn.tree import DecisionTreeClassifier
#choose best max_depth for desicion tree classifier
max_depth_range = list(range(1, 10))
accuracy = []
for depth in max_depth_range:

    clf= DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)
print(accuracy)
#best value is 5 for max_depth

start_time = time.time()
dtc = DecisionTreeClassifier(max_depth = 5)
dtc = dtc.fit(X_train , y_train)
end_time = time.time()
print("")
print('\nDTC Clasification Train time: {}'.format(end_time - start_time))
start_time = time.time()
y_pred1 = dtc.predict(X_test)
end_time = time.time()
print('\nDTC Clasification Predict time: {}'.format(end_time - start_time))
acc = accuracy_score(y_test,y_pred1)
cm = confusion_matrix(y_test,y_pred1)
print('\nDTC Confusion Matrix (without preprocessing):',cm)
print('\nDTC Accuracy Score (without preprocessing):', acc)


#featuring scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)

#dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda = LDA(n_components =1 )
lda = LDA()
X_train2 = lda.fit_transform(X_train1, y_train)
X_test2 = lda.transform(X_test1)

from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=12, p=1)
knn.fit(X_train2,y_train)
end_time = time.time()
print("")
print('\nKNN Train time: {}'.format(end_time - start_time))
start_time = time.time()
y_pred2 = knn.predict(X_test2)
end_time = time.time()
print('\nKNN Predict time: {}'.format(end_time - start_time))
cm = confusion_matrix(y_test,y_pred2)
acc = accuracy_score(y_test,y_pred2)
print('\nKNN Confusion Matrix (with preprocessing):', cm)
print('\nKNN Accuracy Score (with preprocessing):', acc)

#DTC
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
dtc = DecisionTreeClassifier(max_depth = 5)
dtc = dtc.fit(X_train2 , y_train)
end_time = time.time()
print("")
print('\nDTC Train time: {}'.format(end_time - start_time))
start_time = time.time()
y_pred3 = dtc.predict(X_test2)
end_time = time.time()
print('\nDTC predict time: {}'.format(end_time - start_time))
acc = accuracy_score(y_test,y_pred3)
cm = confusion_matrix(y_test,y_pred3)
print('\nDTC Confusion Matrix (with preprocessing):',cm)
print('\nDTC Accuracy Score (with preprocessing):', acc)

import matplotlib.pyplot as plt
from sklearn import tree
feature_names = ['Pregnancies', 'Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_names = ['0', '1']
fig = plt.figure(figsize = (25 , 20))
plot = tree.plot_tree(dtc,
                     feature_names = feature_names,
                     class_names = target_names,
                     filled = True)
fig.savefig('tree.png')


