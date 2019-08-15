import pandas as pd
import numpy as np
from sklearn.tree import  DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
import subprocess
from IPython.display import Image
import pydotplus
import graphviz

from sklearn.externals.six import StringIO
from subprocess import call


import sklearn.metrics as met
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('data/data')

#print(df.head())

features = df.columns[2:].tolist()

x=df[features]
y=df['Edible']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train)
# print(x_test)

parameters = [{'criterion' : ['gini', 'entropy'],
               'max_depth' : range(3, 10),
               'max_leaf_nodes' : range(2, 10),
}]

dt = GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5)

dt.fit(x_train, y_train)

print("Najbolji parametar: ", dt.best_params_)

print("Ocena uspeha po klasifikatorima:")
means = dt.cv_results_['mean_test_score']
stds = dt.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, dt.cv_results_['params']):
    print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))
print()


#Primena modela na test podacima
y_pred = dt.predict(x_test)

print('Matrica konfuzije')
cnf_matrix = met.confusion_matrix(y_test, y_pred)
df_cnf_matrix = pd.DataFrame(cnf_matrix, index = dt.classes_, columns = dt.classes_)
print(df_cnf_matrix)
print()

accuracy = met.accuracy_score(y_test, y_pred, normalize=False)
print('Preciznost u broju instanci', accuracy)

print('Preciznost po klasama', met.precision_score(y_test, y_pred, average=None))

print('Odziv po klasama', met.recall_score(y_test, y_pred, average=None))

class_report = met.classification_report(y_test, y_pred)
print('Izvestaj klasifikacije', class_report, sep='\n')

