# -*- coding: utf-8 -*-
"""
Support Vector Machine introduction
tuto -> https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/

"""


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm

# import data
df = pd.read_csv("data/breast-cancer-wisconsin.txt", sep=',', header=None)

# define column variable name
df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class: (2 for benign, 4 for malignant)']

# delete 'Sample code number' variable
df.drop(df.columns[0], 1, inplace=True)

# delete row with missing variable value
df.replace('?',np.nan, inplace=True)
df.dropna(inplace=True)

# randomize row data order
# note : does sklearn svm integrate shuffle ??? As I'm not sure...
# random_state -> https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Tips on Practical Use -> https://scikit-learn.org/stable/modules/svm.html

df = shuffle(df)

# input variables
X = np.array(df.drop(df.columns[-1], 1))
# output variable
y = np.array(df[df.columns[-1]])

# devide data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialise SVC instance
clf = svm.SVC(gamma='auto')

# train the SMV model
clf.fit(X_train, y_train)

# evaluate precision
confidence = clf.score(X_test, y_test)

# display 1 prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
prediction = clf.predict(example_measures)
print(prediction)



