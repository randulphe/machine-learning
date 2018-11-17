# -*- coding: utf-8 -*-
"""
https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/
"""


import pandas as pd

# import data and define column variable name
df = pd.read_csv("data/breast-cancer-wisconsin.txt", sep=',', header=None)

df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class: (2 for benign, 4 for malignant)']
