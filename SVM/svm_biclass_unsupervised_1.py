# -*- coding: utf-8 -*-
"""
Support Vector Machine introduction
tuto ->     
    code ->     
        https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/
    unsupervised ->
        https://thisdata.com/blog/unsupervised-machine-learning-with-one-class-support-vector-machines/

Here is a really naive implementation of unsupervised SVM
precision is around 80%

for better implementation ->
    define nu and gamma value (nu : percentage of outlier expected, gamma: smoothness of the model)
        nu = y.tolist().count(4) / y.shape[0]
        gamma must be find empirically
    
    see 2nd tuto for more complete precision metrics
    
    save trained model on the disk
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
df = shuffle(df)

# input variables
X = np.array(df.drop(df.columns[-1], 1))
# ground truth output
y = np.array(df[df.columns[-1]])






# devide data into train and test set
X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

# initialise SVC instance
clf = svm.OneClassSVM(gamma='auto')

# train the SMV model
clf.fit(X_train)

# test
example_measures = X_test
prediction = clf.predict(example_measures)

# svm.OneClassSVM() output -1 for outlier and 1 for inlier
# malign 4->-1 and benign 2 -> 1
y_test[y_test==4]=-1
y_test[y_test==2]=1

# +-2 -> good prediction, 0 -> false prediction
test = prediction + y_test
print(test)

# evaluate precision
precision = (test.shape[0]-test.tolist().count(0))/test.shape[0]
print(precision)
