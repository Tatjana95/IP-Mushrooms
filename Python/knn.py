import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.preprocessing as prep
import sklearn.metrics as met
from termcolor import colored
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/data')

features = df.columns[2:].tolist()
#
x=df[features]
y=df['Edible']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [{'n_neighbors': range(1,9),
               'p':[1, 2],
               'weights': ['uniform', 'distance'],
               }]

knn = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)

knn.fit(x_train, y_train)

print("Najbolji parametar: ", knn.best_params_)

print("Ocena uspeha po klasifikatorima:")
means = knn.cv_results_['mean_test_score']
stds = knn.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, knn.cv_results_['params']):
    print("%0.3f (+/-%0.03f) za %s" % (mean, std * 2, params))
print()

y_true, y_pred = y_test, knn.predict(x_test)

print('Matrica konfuzije')
cnf_matrix = met.confusion_matrix(y_test, y_pred)
df_cnf_matrix = pd.DataFrame(cnf_matrix, index = knn.classes_, columns = knn.classes_)
print(df_cnf_matrix)
print()

accuracy = met.accuracy_score(y_test, y_pred, normalize=False)
print('Preciznost u broju instanci', accuracy)

print('Preciznost po klasama', met.precision_score(y_test, y_pred, average=None))

print('Odziv po klasama', met.recall_score(y_test, y_pred, average=None))

class_report = met.classification_report(y_test, y_pred)
print('Izvestaj klasifikacije', class_report, sep='\n')
