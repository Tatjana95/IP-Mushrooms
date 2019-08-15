from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.metrics as met
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/data')

features = df.columns[2:].tolist()

x=df[features]
y=df['Edible']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


mnb = MultinomialNB()
mnb.fit(x_train, y_train)

y_pred = mnb.predict(x_test)

print('Matrica konfuzije')
cnf_matrix = met.confusion_matrix(y_test, y_pred)
df_cnf_matrix = pd.DataFrame(cnf_matrix, index = mnb.classes_, columns = mnb.classes_)
print(df_cnf_matrix)
print()

accuracy = met.accuracy_score(y_test, y_pred, normalize=False)
print('Preciznost u broju instanci', accuracy)

print('Preciznost po klasama', met.precision_score(y_test, y_pred, average=None))

print('Odziv po klasama', met.recall_score(y_test, y_pred, average=None))

class_report = met.classification_report(y_test, y_pred)
print('Izvestaj klasifikacije', class_report, sep='\n')
