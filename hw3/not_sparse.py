
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
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
from imblearn.over_sampling import SMOTE
import argparse

parser = argparse.ArgumentParser(description='Train/Test')
parser.add_argument('csv', metavar='CSV', type=str,
                    help='Path to the csv file')
parser.add_argument('--test', action='store_true', default=False,
                    help='If set, test on new data. (This will override export/import options)')
parser.add_argument('--export_models', action='store_true', default=False,
                    help='If set, output trained models at the end.')
parser.add_argument('--model_dir', type=str, default='./hw3_models' , required=False,
                    help='Root directory to save models / load models')
parser.add_argument('--toy', action='store_true', default=False,
                    help='If set, randomly sample 10000 rows of data.')

args = parser.parse_args()

model_dir = args.model_dir
EXPORT_MODELS=args.export_models
IMPORT_MODELS=False
TESTING=args.test
if TESTING: ## override
    IMPORT_MODELS = True
    EXPORT_MODELS = False


# In[2]:


def load_model(model_path):
    model = None
    with open(str(model_path), 'rb') as handle:
        model = pickle.load(handle)
    return model


# 資料預處理

# In[3]:


from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction import DictVectorizer
train_path = args.csv
if EXPORT_MODELS:
    if not os.path.isdir(model_dir+'/feature_extractors'):
        os.makedirs(model_dir+'/feature_extractors')
    if not os.path.isdir(model_dir+'/pca'):
        os.makedirs(model_dir+'/pca')
    if not os.path.isdir(model_dir+'/models'):
        os.makedirs(model_dir+'/models')

if args.toy:
    data = pd.read_csv(train_path, sep=',', na_values=None, na_filter=False).sample(10000) # toy
else:
    data = pd.read_csv(train_path, sep=',', na_values=None, na_filter=False)

if not TESTING:
    if IMPORT_MODELS and os.path.exists(model_dir+"/feature_extractors/le.pkl"):
        label_le = load_model(model_dir+"/feature_extractors/le.pkl")
    else:
        label_le = LE().fit(data.click)
        if EXPORT_MODELS:
            with open(model_dir+"/feature_extractors/le.pkl", "wb") as f: # export pca transformer
                pickle.dump(label_le, f, pickle.HIGHEST_PROTOCOL)
    label = label_le.transform(data.click)
    del data['click'] # 記得別讓答案變成一組 feature ，這樣 model 就直接看到答案了
selected_col = ['adx','spaceType','spaceId','spaceCat','adType','ip','os','deviceType','publisherId','dclkVerticals','campaignId','advertiserId']
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

if IMPORT_MODELS and os.path.exists(model_dir+"/feature_extractors/dv.pkl"):
    dv = load_model(model_dir+"/feature_extractors/dv.pkl")
else:
    dv = LabelEncoders_fit(data)
    if EXPORT_MODELS:
        with open(model_dir+"/feature_extractors/dv.pkl", "wb") as f: # export pca transformer
            pickle.dump(dv, f, pickle.HIGHEST_PROTOCOL)
data = LabelEncoders_transform(dv, data)

from sklearn.model_selection import train_test_split
if not TESTING:
    data, X_test, label, Y_test = train_test_split(data, np.asarray(label).flatten(), test_size=0.1)
    print data.shape
    print X_test.shape
    print label.shape
    print Y_test.shape


# 設定模型參數、建構模型

# In[7]:

## DT on sklearn is based on CART algorithm. Let's try ID3&CART algorithm both. And SVM.
classifiers = {
                'RandomForest-1-16': RandomForestClassifier(oob_score=True, n_estimators=160, n_jobs=-1, max_depth=5, class_weight={0:1,1:16}),
                # 'LinearSVC': LinearSVC(dual=False, class_weight={0:1,1:16})
              }


# In[8]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# In[9]:

models = []

# 10-fold Cross-validation

# In[12]:


if not TESTING:
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    cvscores = []
    for model_name, model in classifiers.items():
        print('Training %s...'%model_name)
        tempscores = []
        for train, test in kfold.split(data, label):
            X, y = data[train], label[train]
            model.fit(X,y)
            pred = model.predict(data[test])
            pred[pred<0.5] = 0
            pred[pred>0] = 1
            score = f1_score(label[test], pred, average='binary')
            tempscores.append(score)
            models.append((str(model_name), model, score))
        print("f1.avg: %.4f%% (+/- %.4f%%)" % (np.mean(tempscores)*100., np.std(tempscores)*100.)) ## validation 的 f1-score 的平均值 +/- 兩倍標準差
        cvscores.extend(tempscores)


# In[13]:


if IMPORT_MODELS:
    model_dir_model = model_dir+'/models'
    model_list = os.listdir(model_dir_model)
    for filename in model_list:
        full_path = model_dir_model+'/'+filename
        score = float((''.join(filename.split('.')[:-1])).split('-')[-1])
        model = load_model(str(full_path))
        models.append((filename, model, score))


# In[14]:


if EXPORT_MODELS:
    for i, (model_name, model, score) in enumerate(models):
        with open(model_dir+"/models/%02d-%s-%.2f.pkl"%(i, str(model_name), score), "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


# In[16]:


def voter(models, data, tol=0.05): # set tolerance to ignore weak classifier
    final_p = np.zeros(len(data))
    for i, entry in enumerate(models):
        model, score = entry[-2], entry[-1]
        if score<tol: continue
        pred = (np.asarray(model.predict(data)).flatten()-0.5)*score
        final_p += pred # 有權重的投票
    final_p[final_p<0] = 0
    final_p[final_p>0] = 1
    return final_p


# In[25]:


from sklearn.model_selection import train_test_split
if not TESTING:
    predicted = voter(models, X_test, tol=0.05)
    confusion_metrix = skl.metrics.confusion_matrix(Y_test, predicted)
    inclass_precision = skl.metrics.classification_report(Y_test, predicted)
    score = f1_score(Y_test, predicted, average='binary')
    print(confusion_metrix)
    print(inclass_precision)
    print(score)
else:
    predicted = voter(models, data).flatten()
    with open('./results_label.txt', 'w') as fp:
        fp.write('click\n') # field name
        for v in predicted:
            fp.write('%d\n'%int(v))
    print('Answer is in: results_label.txt')


# 補充從 R script 得到的這份 training data 的 ANOVA 結果（從九十萬筆資料隨機抽樣5000筆資料）：
#
# ```R
#                Df Sum Sq  Mean Sq   F value Pr(>F)
# adx             1  0.000 0.000003 2.192e+23 <2e-16 ***
# spaceType       1  0.000 0.000011 9.356e+23 <2e-16 ***
# spaceId       128  0.516 0.004034 3.451e+26 <2e-16 ***
# ip           4839  3.480 0.000719 6.154e+25 <2e-16 ***
# os              3  0.000 0.000000 2.000e-03 0.9999
# campaignId      1  0.000 0.000000 3.089e+00 0.0911 .
# advertiserId    1  0.000 0.000000 1.496e+00 0.2327
# Residuals      25  0.000 0.000000
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# ```
#
# 可以看出 adx, spaceType, spaceId, ip 對 click 的影響似乎是顯著的？
#
# 也許可以實驗看看只選擇 adx, spaceType, spaceId, ip 當作 feature？或是使用 PCA 做降維？
#
# 因為他們對 click 的影響較顯著？ p<0.05
