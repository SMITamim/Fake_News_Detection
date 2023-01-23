import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string

from sklearn.utils import column_or_1d

true = pd.read_csv('LabeledAuthentic-7K.csv')
false = pd.read_csv('LabeledFake-1K.csv')
#print(true)
#print(true.head())
#print(false.head())

#print(true.shape)
#print(false.shape)

true_manual_testing = true.tail(10)
for i in range (7201, 7191, -1):
    true.drop([i], axis=0, inplace=True)
print(true_manual_testing.shape)


false_manual_testing = false.tail(10)
for i in range(1298, 1288, -1):
    false.drop([i], axis=0, inplace=True)
print(false_manual_testing.shape)

#print(false_manual_testing.head(10))


#df_manual_testing = pd.concat([false_manual_testing, true_manual_testing], axis=0)
#df_manual_testing.to_csv("manual_testing.csv")

df_merge = pd.concat([false, true], axis =0 )
#df_merge.head(10)

#df_merge.columns

df = df_merge.drop(["relation", "source", "date"], axis=1)
df.isnull().sum()

df = df.sample(frac=1)
df.head()

df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)
df.drop(["articleID"], axis=1, inplace=True)
df.columns

def wordopt(headline):
    text = headline.lower()
    text = re.sub('\[.*?\]', '', headline)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', headline)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', headline)
    text = re.sub('\n', '', headline)
    text = re.sub('\w*\d\w*', '', headline)
    return headline

df["headline"] = df["headline"].apply(wordopt)
df["content"] = df["content"].apply(wordopt)

x = df["headline"]
y = df["content"]
z = df["label"]

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.25)

##############Convert text to vectors#############

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
yv_train = vectorization.fit_transform(y_train)
yv_test = vectorization.transform(y_test)

df["headline"] = pd.to_numeric(df["headline"])
df["content"] = pd.to_numeric(df["content"])

df['headline'] = df['headline'].str.replace(',', '').astype(float)
df['headline'] = df['headline'].str.replace('-', '').astype(float)
df['headline'] = df['headline'].str.replace('""', '').astype(float)
df['headline'] = df['headline'].str.replace("''", '').astype(float)

df.apply(pd.to_numeric)

df['content'] = df['content'].str.replace(',', '').astype(float)
df['content'] = df['content'].str.replace('-', '').astype(float)
df['content'] = df['content'].str.replace('""', '').astype(float)
df['content'] = df['content'].str.replace("''", '').astype(float)

df.apply(pd.to_numeric)

#################### SVM #############

model_svm = svm.LinearSVC()

model_svm.fit(x_train, y_train, z_train)

z_prediction_svm = model_svm.predict(x_test, y_test)
z = column_or_1d(y, warn=True)

score_svm = metrics.accuracy_score(z_prediction_svm, z_test).round(4)

print('----------Support Vector Machine(SVM)-----------')
print('Accuracy: {}'.format(score_svm))
print('---------------------')

classifier_score_list = set()
classifier_score_list.add(('SVM', score_svm))