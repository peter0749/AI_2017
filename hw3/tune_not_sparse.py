
# coding: utf-8

# In[1]:


# get_ipython().magic(u'matplotlib inline')
import os
import numpy as np
import sklearn as skl
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
from imblearn.over_sampling import SMOTE
import argparse

parser = argparse.ArgumentParser(description='Train/Test')
parser.add_argument('csv', metavar='CSV', type=str,
                    help='Path to the csv file')
parser.add_argument('--toy', action='store_true', default=False,
                    help='If set, randomly sample 8000 rows of data.')

args = parser.parse_args()

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction import DictVectorizer
train_path = args.csv
if args.toy:
    data = pd.read_csv(train_path, sep=',', na_values=None, na_filter=False).sample(8000) # toy
else:
    data = pd.read_csv(train_path, sep=',', na_values=None, na_filter=False)
label_le = LE().fit(data.click)
label = label_le.transform(data.click)
del data['click'] # 記得別讓答案變成一組 feature ，這樣 model 就直接看到答案了
# 特徵選擇、降維 改交給 SVD 分解完成

selected_col = ['adx','spaceType','spaceId','spaceCat','adType','os','deviceType','publisherId','campaignId','advertiserId']
data = data[selected_col]

def LabelEncoders_fit(data):
    le = dict()
    for key, value in data.iteritems():
        le[key] = LE()
        le[key].fit(value)
    return le

def LabelEncoders_transform(le, data):
    result = []
    for key, value in data.iteritems():
        result.append(le[key].transform(value))
    result = np.transpose(np.asarray(result), (1,0))
    return np.asarray(result)

dv = LabelEncoders_fit(data)
data = LabelEncoders_transform(dv, data)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

parameters = {
    'max_depth':[v for v in range(11,14)],
    'min_samples_split': [s for s in range(4,7)],
    'min_samples_leaf': [1],
    'class_weight':[{0:1,1:w} for w in range(11,15)],
    'n_estimators':[40],
    'oob_score':[True,False],
    } ## 想要評估的模型的參數
estimator = RandomForestClassifier() ## 這裡放你想要評估的模型
clf = GridSearchCV(estimator, parameters, n_jobs=-1, scoring='f1', cv=3) ## 多線程執行， 3-fold cross validation
clf.fit(data, label)
print clf.best_score_ ## 最好有多少？
print clf.best_params_ ## 最好的參數

