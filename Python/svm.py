import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as met
from sklearn.svm import  SVC
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('data/data')

features = df.columns[2:].tolist()

x=df[features]
y=df['Edible']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [{'C': [0.01, 0.1, 1],
               'kernel' : ['linear', 'poly', 'sigmoid']

}]

svm = GridSearchCV(SVC(), parameters, cv=5)
svm.fit(x_train, y_train)

print("Najbolji parametri:")
print(svm.best_params_)

print("Ocena uspeha po klasifikatorima:")
means = svm.cv_results_['mean_test_score']
stds = svm.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm.cv_results_['params']):
    print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))
print()


y_true, y_pred = y_test, svm.predict(x_test)

print(met.classification_report(y_true, y_pred))

print(met.confusion_matrix(y_true, y_pred))
