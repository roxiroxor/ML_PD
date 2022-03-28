import sklearn
import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.kernel_ridge import KernelRidge

def convertTuple(tup):
    str = ', '.join(tup)
    return str

df = pd.read_csv("C:\\Data\\parkinson_actigraphy\\dataset_parkinson_diary.csv", encoding='utf-8', sep=";", header=0, index_col=0, decimal=',')


dropped_columns = df.loc[:, df.columns!='Probable Parkinson Disease']

score_saved = 0
min_size = 2
max_size = df.shape[1] - 1
column_subsets = itertools.chain(*map(lambda x: itertools.combinations(dropped_columns, x), range(min_size,max_size+1)))

f = open('C:\\Data\\vysledky_krr_diary.txt', 'w')


for column_subset in column_subsets:
    y = df['Probable Parkinson Disease']
    x = df[list(column_subset)]
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    X_train_scaled = scale(X_train)
    X_test_scaled = scale(X_test)
    optimal_params = KernelRidge(alpha=1.0, kernel='rbf')
    optimal_params.fit(X_train, y_train)
    score_temp = optimal_params.score(X_test, y_test)
    stri = convertTuple(column_subset)
    f.write("%s " %score_temp)
    f.write("%s " %stri)
    f.write("\n")
    if score_temp > score_saved:
        score_saved = score_temp
        best_column = column_subset


print(best_column)
print(score_saved)




param_grid = [
    {'C': [0.5, 1, 10, 100], 'gamma': ['scale', 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]
#optimal_params = GridSearchCV(SVC(),param_grid, cv=5,scoring='accuracy')



#print(optimal_params.best_params_)
