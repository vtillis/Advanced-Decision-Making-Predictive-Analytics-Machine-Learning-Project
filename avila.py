# -*- coding: utf-8 -*-
"""avila.ipynb

## Avila Bible
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

"""Download and extract data"""

! wget -O avilia.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip
! unzip avilia.zip

names = ['intercolumnar_distance', 'upper_margin', 'lower_margin', 'exploitation', 'row_number', 'modular_ratio',
         'interlinear_spacing', 'weight', 'peak_number', 'modular_ratio_interlinear_spacing', 'class']
df = pd.read_csv('avila/avila-tr.txt', names=names)
df.head()

"""Exploratory Data Analysis"""

# data description and information

print('Instances:', df.shape[0])
print('Features:', df.shape[1])

print('\nInfo:')
df.info()

print('\nDescription:')
df.describe()

# check for null values

print('\nNULL:')
print(df.isnull().any())

# class distribution

df['class'] = pd.factorize(df['class'], sort=True)[0] # convert class labels from letters to ints
print(df['class'].value_counts().sort_index())

plt.figure(figsize=(12, 6))
df['class'].value_counts().sort_index().plot(kind='bar')

# data distrinution

df.hist(figsize=(16, 8), layout=(2, 6), bins = 50)

# data feature correlation 1

sns.pairplot(df)

# data feature correlation 2

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True)

"""Preprocessing"""

# outliers

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient='h')


df = df[np.abs(df.upper_margin - df.upper_margin.mean()) <= (15 * df.upper_margin.std())]
# keep only the ones that are within +15 to -15 standard deviations in the column 'Data'.

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient='h')

df.info()

# split train data to features and labels
X_train = df.drop('class', axis = 1)
y_train = df['class']

# import test data
df_test = pd.read_csv('avila/avila-ts.txt', names=names)
df_test['class'] = pd.factorize(df_test['class'], sort=True)[0] # convert class labels from letters to ints
print(df_test['class'].value_counts().sort_index())

plt.figure(figsize=(12, 6))
df_test['class'].value_counts().sort_index().plot(kind='bar')

# split test data to features and labels
X_test = df_test.drop('class', axis = 1)
y_test = df_test['class']

from sklearn.preprocessing import StandardScaler

# data scaling
scaler = StandardScaler()
scaler.fit_transform(X_train) # fit scaler only on traing
scaler.transform(X_test)

"""Classification"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

accs =[]
accs_without_opt =[]
names2 =[]
times2 =[]

from sklearn.naive_bayes import GaussianNB

params = {'var_smoothing': np.linspace(1e-10, 1e-8, 5)}

gnb = GaussianNB()   
clf = GridSearchCV(gnb, params, scoring=('balanced_accuracy'))
clf.fit(X_train, y_train)

print(clf.best_params_)

start_time = time.time()
best_clf = clf.best_estimator_
best_clf.fit(X_train, y_train)

acc = best_clf.score(X_test, y_test)
accs.append(acc)
names2.append('Naive Bayes')

t = time.time() - start_time
times2.append(t)
print("Συνολικός χρόνος fit και predict: %s seconds" % (t))
print('Accuracy', acc) 

clf_s = gnb
clf_s.fit(X_train, y_train)
acc = clf_s.score(X_test, y_test)
accs_without_opt.append(acc)

print('Without Optimization Accuracy', acc)

y_pred = best_clf.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n')
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')

train_acc = best_clf.score(X_train, y_train) 
print('Training Score: ', train_acc)
test_acc = best_clf.score(X_test, y_test)
print('Testing Score: ', test_acc)

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

params = {'criterion': ['gini', 'entropy'],
          'min_samples_split': np.arange(1,5),
          'min_samples_leaf': np.arange(1,5)
}

rfc = RandomForestClassifier()  
clf = GridSearchCV(rfc, params, scoring=('balanced_accuracy'))
clf.fit(X_train, y_train)

print(clf.best_params_)

start_time = time.time()
best_clf = clf.best_estimator_
best_clf.fit(X_train, y_train)

acc = best_clf.score(X_test, y_test)
accs.append(acc)
names2.append('Random Forest')

t = time.time() - start_time
times2.append(t)
print("Συνολικός χρόνος fit και predict: %s seconds" % (t))
print('Accuracy', acc) 

clf_s = rfc
clf_s.fit(X_train, y_train)
acc = clf_s.score(X_test, y_test)
accs_without_opt.append(acc)

print('Without Optimization Accuracy', acc)

y_pred = best_clf.predict(X_test)

print('Classification Report: \n', classification_report(y_test, y_pred))
print('Confusion Matrix: \n')
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')

train_acc = best_clf.score(X_train, y_train) 
print('Training Score: ', train_acc)
test_acc = best_clf.score(X_test, y_test)
print('Testing Score: ', test_acc)

plt.figure(figsize=(10,5))

plt.barh(names2, accs)
plt.ylabel('Algorithm')
plt.xlabel('Accuracy')
plt.title('Algorithms and correspoding Accuracy')

plt.figure(figsize=(10,5))

plt.barh(names2, times2)
plt.ylabel('Algorithm')
plt.xlabel('Training and Prdiction Time')
#plt.xlim(0.5, 1.0)
plt.title('Algorithms and correspoding Execution Time')
