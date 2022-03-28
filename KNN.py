import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier


#Merging strings
def convertTuple(tup):
    str = ', '.join(tup)
    return str

df = pd.read_csv("C:\\Data\\parkinson_actigraphy\\dataset_parkinsonMCI_diary.csv", encoding='utf-8', sep=";", header=0, index_col=0, decimal=',')


#Columns used for iteration (input values x except for value y -> Probable Parkinsons Disease)
iter_columns = df.loc[:, df.columns!='Probable Parkinson Disease']

#Setting up iterator for combinations

score_saved = 0
min_size = 2
max_size = df.shape[1]
column_subsets = itertools.chain(*map(lambda x: itertools.combinations(iter_columns, x), range(min_size,max_size)))

#Creating csv file for writing results of individual SVMs
f = open('C:\\Data\\vysledkyKNN_diaryPDMCI.csv', 'w')

text = df.columns
print(text)
y = df['Probable Parkinson Disease']
x = df.loc[:, df.columns!='Probable Parkinson Disease']
param_grid = [
    {'n_neighbors': [4, 5, 6], 'weights': ['uniform','distance']},
]

for column_subset in column_subsets:
    #Splitting current data and fitting them into SVM
    y = df['Probable Parkinson Disease']
    x = df[list(column_subset)]
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)

    optimal_params = GridSearchCV(KNeighborsClassifier(),param_grid, cv=5,scoring='accuracy')
    optimal_params.fit(X_train, y_train)
    score_temp = optimal_params.score(X_test, y_test)
    stri = convertTuple(column_subset)

    # Writing results into file
    f.write("%s;" % score_temp)
    f.write("%s;" % optimal_params.best_params_)
    f.write("%s" % stri)
    f.write("\n")