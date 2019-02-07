
from IPython.core.display import display, HTML
from sklearn.datasets import load_iris
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import random
from scipy.stats import randint as sp_randint
import pickle

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score, cross_validate

from keras.models import Sequential
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import SGD
from utils import param_search_report, create_nn_model, param_search

random_state = 100

## DATASET PREP
heart_colnames = ['age','sex','chest_pain_type',
                  'resting_bp','serum_chol','fasting_bs',
                  'resting_ecg','max_hr','angina','oldpeak',
                  'peak_st_slope','num_vessels','thal','heart_disease']
abalone_colnames = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']

## LOAD DATA
heart = pd.read_table('/Users/Andy/Documents/OMSCS/machine_learning/heart.txt', sep=' ', header=None, names=heart_colnames)
abalone = pd.read_table('/Users/Andy/Documents/OMSCS/machine_learning/abalone.txt', sep=',',header=None, names=abalone_colnames)
target_counts = abalone['rings'].value_counts()
target_counts.sort_index(inplace=True)
target_counts.cumsum()
abalone['target'] = np.where(abalone['rings'] > 7, 0, 1)
abalone = pd.get_dummies(data=abalone, prefix='sex') #craete dummies for sex
heart['target'] = np.where(heart['heart_disease'] == 1, 0, 1)
heart.drop('heart_disease',axis=1,inplace=True)
abalone.drop('rings', axis=1, inplace=True)

## HEART - SEP. TARGET AND FEATURES
heart_x = heart.iloc[:,:-1].values
heart_x = scaler.fit_transform(heart_x)
heart_y = heart.iloc[:,-1].values

## ABALONE - SEP. TARGET AND FEATURES
abalone_x = abalone.drop('target',axis=1).values
abalone_x = scaler.fit_transform(abalone_x)
abalone_y = abalone['target'].values

## HEART - TRAIN TEST SPLIT
x_train_heart, x_test_heart, y_train_heart, y_test_heart = \
    train_test_split(heart_x, heart_y, test_size = 0.3, random_state = random_state)

## ABALONE - TRAIN TEST SPLIT
x_train_abalone, x_test_abalone, y_train_abalone, y_test_abalone = \
    train_test_split(abalone_x, abalone_y, test_size = 0.3, random_state = random_state)

### CREATE BASELINE PREDICTIONS AND OUTPUT CSV WITH RESULTS
for dataset in [(x_train_heart, y_train_heart, x_test_heart, y_test_heart, 'heart'), (x_train_abalone, y_train_abalone, x_test_abalone, y_test_abalone, 'abalone')]:
    x_train = dataset[0]
    y_train = dataset[1] 
    x_test = dataset[2]
    y_test = dataset[3]
    dataset = dataset[4]
    
    baseline_mod = DummyClassifier(random_state = random_state)
    baseline_mod.fit(x_train, y_train)
    pred = baseline_mod.predict(x_test)
    pred_proba_positive = baseline_mod.predict_proba(x_test)
    pred_proba_positive = pred_proba_positive[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    matrix = pd.DataFrame(data = {'Predicted Negative':[tn, fn], 'Predicted Positive':[fp, tp]},index = ['Negative', 'Positive'])
    matrix.to_csv(dataset + 'baseline.csv')
    
    acc = baseline_mod.score(x_test, y_test)
    apr = average_precision_score(y_test, pred)
#     r = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba_positive)
    
    metrics = pd.DataFrame(data = {'Metric': ['Accuracy','APR', 'AUC'],
                                   'Value': [acc, apr, auc]})
    print('====================================================================')
    print('baseline prediction for {0}'.format(dataset))
    display(metrics)
    metrics.to_csv(dataset+'_baseline_metrics.csv', index=False)

DECISION TREE CLASSIFIER ###
ABALONE ###

abalone_dt = DecisionTreeClassifier(random_state=random_state)

param_dist = {
    'criterion' : ['gini','entropy'],
    'max_depth' : [1, 2, 3, 4, 5, 10, 15, 20, 25,30], 
}

abalone_dt, cv_results = param_search(x_train_abalone, 
                                      y_train_abalone, 
                                      search_type = 'Grid', 
                                      param_dist = param_dist, 
                                      model = abalone_dt,
                                      scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - DECISION TREE - ABALONE')
param_search_report(cv_results, n_top = 100, save_as = 'abalone_dt_param_report', print_n_top=10, verbose=True)
pickle.dump(abalone_dt, open("abalone_dt.p","wb"))

heart_dt = DecisionTreeClassifier(random_state=random_state)
param_dist = {
    'criterion' : ['gini','entropy'],
    'max_depth' : [1, 2, 3, 4, 5, 10, 15, 20, 25, 30], 
}
heart_dt, cv_results = param_search(x_train_heart, 
                                    y_train_heart, 
                                    search_type = 'Grid', 
                                    param_dist = param_dist, 
                                    model = heart_dt,
                                    scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - DECISION TREE - HEART')
param_search_report(cv_results, n_top = 100, save_as = 'heart_dt_param_report', verbose=True, print_n_top=10)
pickle.dump(heart_dt, open("heart_dt.p","wb"))

### NN Classifier
### abalone ### 
abalone_nn = KerasClassifier(build_fn=create_nn_model,
                             input_dim=10, 
                             verbose=0, 
                             epochs = 50, 
                             batch_size = 100,
                             loss = 'binary_crossentropy')
param_dist = {'layer1_nodes': [5,10,20,25],
              'layer2_nodes': [5,10,20,25],
              'learn_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
              'momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
             }
abalone_nn, cv_results = param_search(x_train_abalone, 
                                      y_train_abalone,
                                      search_type = 'Random',
                                      param_dist = param_dist,
                                      iterations = 100,
                                      model = abalone_nn, 
                                      verbose = True,
                                      scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - NEURAL NETWORK - ABALONE')
param_search_report(cv_results, n_top=100, save_as = 'abalone_nn_param_search', verbose=True, print_n_top=10)
pickle.dump(abalone_nn, open("abalone_nn.p","wb"))

### heart ##
heart_nn = KerasClassifier(build_fn=create_nn_model,
                           input_dim=13, 
                           verbose=0, 
                           epochs=50, 
                           batch_size=100,
                           loss = 'binary_crossentropy')
param_dist = {'layer1_nodes': [5,10,20,25],
              'layer2_nodes': [5,10,20,25],
              'learn_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
              'momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
             }
heart_nn, cv_results = param_search(heart_x, 
                                    heart_y,
                                    search_type = 'Random',
                                    param_dist = param_dist, 
                                    model = heart_nn, 
                                    iterations = 100, 
                                    verbose = True,
                                    scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - NEURAL NETW0RK - HEART')
param_search_report(cv_results, n_top=100, save_as = 'heart_nn_param_search', verbose=True, print_n_top=10)
pickle.dump(heart_nn, open("heart_nn.p","wb"))

### Boosted Classifier
### abalone ##
abalone_ada = AdaBoostClassifier(random_state=random_state)
param_dist = {'n_estimators' : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,150, 200],
             'learning_rate' : [0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2]}
abalone_ada, cv_results = param_search(x_train_abalone, 
                                       y_train_abalone, 
                                       search_type = 'Grid',
                                       param_dist=param_dist, 
                                       model = abalone_ada,  
                                       verbose=True,
                                       scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - ADA BOOST - ABALONE')
param_search_report(cv_results, n_top=100, save_as = 'abalone_ada_param_search', verbose=True, print_n_top=10)
pickle.dump(abalone_ada, open("abalone_ada.p","wb"))

### heart ##
heart_ada = AdaBoostClassifier(random_state=random_state)
param_dist = {'n_estimators' : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
             'learning_rate' : [0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2]}
heart_ada, cv_results = param_search(x_train_heart,
                                     y_train_heart,
                                     search_type = 'Grid',
                                     param_dist = param_dist,
                                     model = heart_ada, 
                                     verbose = True,
                                     scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - ADA BOOST - HEART')
param_search_report(cv_results, n_top=100, save_as = 'heart_ada_param_search', verbose=True, print_n_top=10)
pickle.dump(heart_ada, open("heart_ada.p","wb"))

### KNN Classifier
### Abalone ##
abalone_knn = KNeighborsClassifier(weights = 'uniform', n_jobs = -1)
param_dist = {'n_neighbors': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30], 
              'metric': ['euclidean','manhattan','chebyshev']}
abalone_knn, cv_results = param_search(x_train_abalone,
                                       y_train_abalone,
                                       param_dist = param_dist,
                                       model = abalone_knn,
                                       search_type = 'Grid',
                                       verbose= True,
                                       scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - KNNEIGHBOR - ABALONE')
param_search_report(cv_results, n_top=100, save_as = 'abalone_knn_param_search', verbose=True, print_n_top=10)
pickle.dump(abalone_knn, open("abalone_knn.p","wb"))

### heart ##
heart_knn = KNeighborsClassifier(weights = 'uniform', n_jobs = -1)
param_dist = {'n_neighbors': [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30], 
              'metric': ['euclidean','manhattan','chebyshev']}
heart_knn, cv_results = param_search(x_train_heart,
                                     y_train_heart,
                                     param_dist = param_dist,
                                     search_type = 'Grid',
                                     model = heart_knn,
                                     verbose= True,
                                     scoring = 'roc_auc')
print('====================================================================')
print('PARAM SEARCH REPORT - KNNEIGHBOR - HEART')
param_search_report(cv_results, n_top=100, save_as = 'heart_knn_param_search', verbose=True, print_n_top=10)
pickle.dump(heart_knn, open("heart_knn.p","wb"))

## ABALONE SUPPORT VECTOR MACHINE ##
## abalone ##
abalone_svm = SVC(random_state = random_state, probability=True)

param_dist = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

abalone_svm, cv_results = param_search(x_train_abalone,
                                       y_train_abalone,
                                       param_dist = param_dist,
                                       search_type = 'Grid',
                                       model = abalone_svm,
                                       verbose = True,
                                       scoring = 'roc_auc')

print('====================================================================')
print('PARAM SEARCH REPORT - SVM - ABALONE')
param_search_report(cv_results, n_top=100, save_as = 'abalone_svm_param_search', verbose=True, print_n_top=10)
pickle.dump(abalone_svm, open("abalone_svm.p","wb"))

## heart ##
heart_svm = SVC(random_state = random_state, probability=True)

param_dist = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

heart_svm, cv_results = param_search(x_train_heart,
                                     y_train_heart,
                                     param_dist = param_dist,
                                     search_type = 'Grid',
                                     model = heart_svm,
                                     verbose= True,
                                     scoring = 'roc_auc')

print('====================================================================')
print('PARAM SEARCH REPORT - SVM - HEART')
param_search_report(cv_results, n_top=100, save_as = 'heart_svm_param_search', verbose=True, print_n_top=10)
pickle.dump(heart_svm, open("heart_svm.p","wb"))
