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
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import unicodeconverter as uc
#from PyBanglaPhonetics import phonetics
#from transliterate import translit




from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import column_or_1d


true = pd.read_csv('LabeledAuthentic-7K.csv')
false = pd.read_csv('LabeledFake-1K.csv')
#print(true)
#print(true.head())
#print(false.head())

#print(true.shape)
#print(false.shape)








true_manual_testing = true.tail(10)
for i in range (270, 260, -1):
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
    headline = headline.lower()
    headline = re.sub('\[.*?\]', '', str(headline))
    headline = re.sub("\\W"," ",str(headline))
    headline = re.sub('https?://\S+|www\.\S+', '',str(headline))
    headline = re.sub('<.*?>+', '', str(headline))
    headline = re.sub('[%s]' % re.escape(string.punctuation), '', str(headline))
    headline = re.sub('\n', '',str(headline))
    headline = re.sub('\w*\d\w*', '', str(headline))
    headline = (re.sub('.', lambda x: r'\u % 04X' % ord(x.group()), str(headline)))
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

"""
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


#################### DecisionTree #####################

model_decision_tree = DecisionTreeClassifier(random_state=4)

model_decision_tree.fit(x_train, y_train, z_train)
z_prediction = model_decision_tree.predict(x_test, y_test)

score_decision_tree = metrics.accuracy_score(z_prediction, y_test).round(4)

print('----------Decision Tree-----------')
print('Accuracy: {}'.format(score_decision_tree))
print('---------------------')

classifier_score_list = set()
classifier_score_list.add(('DT', score_decision_tree))


import pandas as pd
from sklearn.model_selection import train_test_split
import re, string, collections, os, random
import sys
sys.path.insert(1, '../helper/')
import helper



df = false
labels = []
for row in df.iterrows():
    row = row[1]
    label = row["label"]
    labels.append(label)


y_train, y_test = train_test_split(labels, test_size=0.3, random_state=109)
true_df = []
fake_df = []
overall_df = []

def random_baseline():
    y_random = []

    for i in range(len(y_test)):
        b = random.choice([0, 1])
        y_random.append(b)
    t,f,o = helper.getReport(y_test, y_random)
    true_df.append(t)
    fake_df.append(f)
    overall_df.append(o)




y_major = []


for i in range(len(y_test)):
    a = 1
    b = random.choice([0, 1])
    y_major.append(a)


for i in range(10):
    random_baseline()



print("Major Baseline")
t,f,o = helper.getResult(y_test, y_major)



print("Fake")
df_result = pd.DataFrame(fake_df)
mean = df_result.mean(axis=0)
random_fake=mean.to_dict()

print("overall")
df_result = pd.DataFrame(overall_df)
mean = df_result.mean(axis=0)
random_over=mean.to_dict()

print("                                Overall               #               Fake                ")
print("                   precision    recall      f1-score  #  precision    recall      f1-score")
helper.printResult("Majority",o,f)
helper.printResult("Random",random_over,random_fake)
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import string
import pickle
import sys
import helper


sys.path.insert(1, '../Helper/')
import helper, config


def getStopWords():
    stopWords = []
    with open(config.STOP_WORD, 'r', encoding="utf8") as f:
        for row in f:  # iterate over the rows in the file
            row = row.replace("\n", "")
            stopWords.append(row)

    return stopWords


def tokenizer(doc):
    puncList = ["।", "”", "“", "’"]
    for p in string.punctuation.lstrip():
        puncList.append(p)
    # remove punctuation
    tokens = []

    def cleanword(word):
        for p in puncList:
            word = word.replace(p, "")
        word = re.sub(r'[\u09E6-\u09EF]', "", word, re.DEBUG)  # replace digits

        return word

    for word in doc.split(" "):
        word = cleanword(word)
        if word != "":
            tokens.append(word)

    return tokens


def word_emb(size, X):
    vocab = set()
    vector = {}

    we300 = config.EMBEDDING_300
    we100 = config.EMBEDDING_100

    if size == 100:
        w = we100
    else:
        w = we300

    with open(w, 'r', encoding="utf8") as file:
        for _, line in enumerate(file):
            l = line.split(' ')
            v = [float(i) for i in l[1:]]
            w = l[0]
            vector[w] = np.array(v)
            vocab.add(w)

    print("Vocab Size =>%s" % (len(vocab)))

    def doc2MeanValue(doc):
        tokens = tokenizer(doc)
        tokentovaluelist = [vector[token] for token in tokens if token in vocab]
        return np.array(tokentovaluelist)

    df = X

    featureVector = []
    labels = []
    for row in df.iterrows():
        row = row[1]
        # id = row["articleID"]
        mean = doc2MeanValue(row["content"])
        if mean.size == 0:
            mean = [0] * size
            featureVector.append(mean)
            continue
        mean = np.mean(mean, axis=0)
        label = row["label"]
        mean = (mean.tolist())
        labels.append(label)
        featureVector.append(mean)

    df = pd.DataFrame(featureVector)
    df = df.fillna(0)
    return sparse.csr.csr_matrix(df.values)


def tfidf_charF(X, a, b, save_model=False):
    X = X.content
    tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b), stop_words=getStopWords(),
                                 decode_error='replace', encoding='utf-8', analyzer='char')

    tfidf_char.fit(X.values.astype('U'))
    #     if save_model:
    #         name = "tfidf_char.pkl"
    #         path = config.API+name
    #         outfile = open(path, 'wb')
    #         pickle.dump(tfidf_char, outfile)
    #         outfile.close()
    x_char = tfidf_char.transform(X.values.astype('U'))
    return x_char


def tfidf_wordF(X, a, b):
    X = X.content
    tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b),
                                 stop_words=getStopWords(), decode_error='replace',
                                 encoding='utf-8', analyzer='word', tokenizer=tokenizer)

    tfidf_word.fit(X.values.astype('U'))

    #     if save_model:
    #         name = "tfidf_word.pkl"
    #         path = config.API+name
    #         outfile = open(path, 'wb')
    #         pickle.dump(tfidf_char, outfile)
    #         outfile.close()

    x_word = tfidf_word.transform(X.values.astype('U'))
    return x_word


def mp(X):
    def count_punc(content):
        char_list = list(content)
        count = 0
        for c in char_list:
            if c == '!':
                count += 1
        return count

    def load_rank():
        rank = {}
        mx = 0
        na = []
        paths = os.listdir(config.ALEXA_RANK)
        for path in paths:
            d = pd.read_csv(config.ALEXA_RANK + "/" + path)
            # d = d.drop(['Unnamed: 0'], axis=1)
            d = d.fillna(0)
            # print(d[d['rank'].isnull()])
            for row in d.iterrows():
                row = row[1]
                if row['rank'] == 0:
                    na.append(row['domain'])
                else:
                    rank[row['domain']] = int(row['rank'])
                    mx = max(int(row['rank']), mx)
        for n in na:
            rank[n] = mx
        return rank

    ranks = load_rank()
    df = X
    featureVector = []
    for row in df.iterrows():
        row = row[1]
        feature = []
        feature.append(count_punc(row['content']))
        r = ranks[row['domain']]
        feature.append(r)
        featureVector.append(feature)

    dfMP = pd.DataFrame(featureVector)
    normalized_df = (dfMP - dfMP.mean()) / dfMP.std()
    dfMP = normalized_df.fillna(0)
    return sparse.csr.csr_matrix(dfMP.values)


def pos():
    path_pos = config.POS_PATH
    dfPOS = pd.read_csv(path_pos)
    dfPOS = dfPOS.fillna(0)
    dfPOS = dfPOS.drop(['articleID', 'Unnamed: 0'], axis=1)
    X_POS = sparse.csr.csr_matrix(dfPOS.values)
    return X_POS
