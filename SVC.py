import sklearn
import pandas as pd
import numpy as np
import itertools
import imblearn

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

#Merging strings
def convertTuple(tup):
    str = ', '.join(tup)
    return str

#CSV reader
df = pd.read_csv("C:\\Data\\parkinson_actigraphy\\dataset_parkinson_diary.csv", encoding='utf-8', sep=";", header=0, index_col=0, decimal=',')

param_grid = [
    {'C': [0.5, 1, 10, 50, 100], 'gamma': ['scale', 10, 1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
]
iter = 0

#Columns used for iteration (input values x except for value y -> Probable Parkinsons Disease)
iter_columns = df.loc[:, df.columns!='Probable Parkinson Disease']

#Setting up iterator for combinations
score_saved = 0
min_size = 2
max_size = df.shape[1]
column_subsets = itertools.chain(*map(lambda x: itertools.combinations(iter_columns, x), range(min_size,max_size)))

#Creating csv file for writing results of individual SVMs
f = open('C:\\Data\\vysledky_diary.csv', 'w')

#Imblearn setup
sampling = SMOTE(random_state=42)

#Iteration
for column_subset in column_subsets:
    #Splitting current data and fitting them into SVM
    y = df['Probable Parkinson Disease']
    x = df[list(column_subset)]
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    # X_resampled, y_resampled = sampling.fit_resample(X_train, y_train)

    optimal_params = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    optimal_params.fit(X_train, y_train)
    score_temp = optimal_params.score(X_test, y_test)
    stri = convertTuple(column_subset)

    #Writing results into file
    f.write("%s;" %score_temp)
    f.write("%s;" %optimal_params.best_params_)
    f.write("%s" %stri)
    f.write("\n")
    iter = iter + 1
    print(iter)




# if score_temp > score_saved:
#     score_saved = score_temp
#     best_column = column_subset
#     disp = ConfusionMatrixDisplay.from_estimator(optimal_params, X_test, y_test)
#
# print(best_column)
# print(score_saved)
#
# plt.title(best_column, fontdict={'fontsize': 11})
# plt.show()


#optimal_params = GridSearchCV(SVC(),param_grid, cv=5,scoring='accuracy')



#print(optimal_params.best_params_)


