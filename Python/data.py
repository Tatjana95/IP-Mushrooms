import pandas as pd
import  numpy as np
from sklearn.preprocessing import LabelEncoder

with open('data/expanded') as f:
    lines = f.readlines()

datas = []

for i in range(9, 8425):
    datas.append(lines[i].replace('\n', '').split(sep = ','))

df = pd.DataFrame(datas, columns=['Edible', 'Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment', 'Gill-spacing', 'Gill-size', 'Gill-color', 'Stalk-shape',
                                  'Stalk-root', 'Stalk-surface-above-ring', 'Stalk-surface-below-ring', 'Stalk-color-above-ring', 'Stalk-color-below-ring', 'Veil-type',
                                  'Veil-color', 'Ring-number', 'Ring-type', 'Spore-print-color', 'Population', 'Habitat'])

print(df.describe())

print()
print('Udeo jestivih i otrovnih pecuraka:')
print(df.groupby('Edible').size())
print(df.groupby('Edible').size() / df['Edible'].count())

print()

df['Edible'] = df['Edible'].replace(['EDIBLE', 'POISONOUS'], [1, 0])
df['Bruises'] = df['Bruises'].replace(['BRUISES', 'NO'], [1, 0])

total_rows = df.shape[0]
#print(total_rows)
df = df.replace('?', np.nan)
print(df.isna().sum() / total_rows)

df = df.drop(['Stalk-root', 'Veil-type'], 1)


df = pd.get_dummies(df)


print(df.head())

df.to_csv('data/data')
