
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
from utils import param_search_report, create_nn_model, param_search, plot_learning_curve, gridsearch_heatmap, param_boxplots

heart_dt_params = pd.read_csv('heart_dt_param_report.csv')
abalone_dt_params = pd.read_csv('abalone_dt_param_report.csv')

heart_nn_params = pd.read_csv('heart_nn_param_search.csv')
abalone_nn_params = pd.read_csv('abalone_nn_param_search.csv')

heart_ada_params = pd.read_csv('heart_ada_param_search.csv')
abalone_ada_params = pd.read_csv('abalone_ada_param_search.csv')

heart_knn_params = pd.read_csv('heart_knn_param_search.csv')
abalone_knn_params = pd.read_csv('abalone_knn_param_search.csv')

heart_svm_params = pd.read_csv('heart_svm_param_search.csv')
abalone_svm_params = pd.read_csv('abalone_svm_param_search.csv')

param_reports = [heart_dt_params, abalone_dt_params,
                 heart_nn_params, abalone_dt_params, 
                 heart_ada_params, abalone_ada_params,
                 heart_knn_params, abalone_ada_params,
                 heart_svm_params, abalone_svm_params]

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

print('++++++++++++++ DECISION TREE ANALYSIS SECTION ++++++++++++++++')
arr = heart_dt_params.pivot(columns='criterion',index='max_depth', values='test_score')
gridsearch_heatmap(arr, 
                   xlab = 'Criterion', 
                   ylab = 'Max Depth',
                   title = 'Heart DT GridSearch',
                   save_as='heart_dt_param_search_chart',
                   xtick_marks=['entropy','gini'])

arr = abalone_dt_params.pivot(columns='criterion', index = 'max_depth', values = 'test_score')
gridsearch_heatmap(arr,
                   xlab = 'Criterion',
                   ylab = 'Max Depth',
                   title = 'Abalone DT GridSeach',
                   save_as = 'abalone_dt_param_search_chart')

print("TEST DIFFERENT TYPES OF PRUNING APPROACHES FOR THE DECISION TREE")
def test_pruning(param, param_list):
    abalone_dt = DecisionTreeClassifier(random_state=random_state)
    param_dist = {
        'criterion' : ['entropy','gini'],
        param : param_list, 
    }
    abalone_dt, cv_results = param_search(x_train_abalone, 
                                           y_train_abalone, 
                                           search_type = 'Grid', 
                                           param_dist = param_dist, 
                                           model = abalone_dt,
                                           scoring = 'roc_auc')
    print('====================================================================')
    print('PARAM SEARCH REPORT - DECISION TREE - ABALONE')
    param_search_report(cv_results, n_top = 3, save_as = 'PRUNING_abalone_'+param, print_n_top=3, verbose=True)

    heart_dt = DecisionTreeClassifier(random_state=random_state)
    param_dist = {
        'criterion' : ['entropy','gini'],
        param : param_list, 
    }
    heart_dt, cv_results = param_search(x_train_heart, 
                                        y_train_heart, 
                                        search_type = 'Grid', 
                                        param_dist = param_dist, 
                                        model = heart_dt,
                                        scoring = 'roc_auc')
    print('====================================================================')
    print('PARAM SEARCH REPORT - DECISION TREE - HEART')
    param_search_report(cv_results, n_top = 3, save_as = 'PRUNING_heart_'+param, verbose=True, print_n_top=3)

    
test_pruning('min_samples_split', [10, 20, 40, 60, 80, 100, 150, 200])
test_pruning('min_samples_leaf', [5,10, 20, 40, 60, 80, 100, 150, 200])
test_pruning('max_leaf_nodes', [10, 20, 40, 60, 80, 100, 120, 140,160,200])




print('++++++++++ ANN ANALYSIS SECTION ++++++++++++++')
param_boxplots(heart_nn_params, problem = 'heart', plot_type=sns.violinplot)

param_boxplots(abalone_nn_params, problem = 'abalone', plot_type=sns.swarmplot)

heart_nn = pickle.load(open('heart_nn.p','rb'))
epochs = list(range(50, 500, 25))
test_avgs = []
train_avgs = []
for num in epochs:
    print('''Starting Run with Epochs = {0}'''.format(num))
    heart_nn.set_params(epochs = num)
    cv_results = cross_validate(heart_nn, heart_x, heart_y, cv = 3, n_jobs=-1, scoring = 'roc_auc')
    print(cv_results.keys())
    test_avgs.append(np.average(cv_results['test_score']))
    train_avgs.append(np.average(cv_results['train_score']))
print(test_avgs)
print(train_avgs)
plt.plot(epochs, test_avgs, color = 'r',label = 'test averages')
plt.plot(epochs, train_avgs, label = 'train averages')
plt.xlabel('epochs')
plt.ylabel('AUC')
plt.legend(loc = 'best')
plt.savefig('Heart NN Epochs.png')





print('++++++++++ ADABoost ANALYSIS SECTION ++++++++++++++')
arr = heart_ada_params.pivot(columns='learning_rate',index='n_estimators', values='test_score')
gridsearch_heatmap(arr, 
                   xlab = 'Learning Rate', 
                   ylab = 'Num Estimators',
                   title = 'Heart ADA GridSearch',
                   save_as='heart_ada_param_search_chart')

arr = abalone_ada_params.pivot(columns='learning_rate',index='n_estimators', values='test_score')
gridsearch_heatmap(arr, 
                   xlab = 'Learning Rate', 
                   ylab = 'Num Estimators',
                   title = 'Abalone ADA GridSearch',
                   save_as='abalone_ada_param_search_chart')

abalone_ada = pickle.load(open('abalone_ada.p','rb'))
estimators = list(range(50, 500, 25))
test_avgs = []
train_avgs = []
for num in estimators:
    print('''Starting Run with n_estimators = {0}'''.format(num))
    abalone_ada.set_params(n_estimators = num)
    cv_results = cross_validate(abalone_ada, abalone_x, abalone_y, cv = 3, n_jobs=-1, scoring = 'roc_auc')
    test_avgs.append(np.average(cv_results['test_score']))
    train_avgs.append(np.average(cv_results['train_score']))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(estimators, test_avgs, color = 'r',label = 'test averages')
ax1.set_ylabel('Test AUC Averages')
ax1.set_xlabel('Epochs')
for tl in ax1.get_yticklabels():
    tl.set_color('r')

ax2 = ax1.twinx()
ax2.plot(estimators, train_avgs, label = 'train averages')
ax2.set_ylabel('Train AUC Averages')
plt.tight_layout()
fig.legend(loc='best')
for tl in ax2.get_yticklabels():
    tl.set_color('b')
plt.savefig('abalone ada N estimators analysis.png')

print('++++++++++ KNN ANALYSIS SECTION ++++++++++++++')
arr = heart_knn_params.pivot(columns='metric',index='n_neighbors', values='test_score')
gridsearch_heatmap(arr, 
                   xlab = 'Distance Metric', 
                   ylab = 'Num. Neighbors',
                   title = 'Heart KNN GridSearch',
                   save_as='heart_knn_param_search_chart',
                   rotation = 90)

arr = abalone_knn_params.pivot(columns='metric',index='n_neighbors', values='test_score')
gridsearch_heatmap(arr, 
                   xlab = 'Distance Metric', 
                   ylab = 'Num. Neighbors',
                   title = 'Abalone KNN GridSearch',
                   save_as='abalone_knn_param_search_chart',
                   rotation = 90)


print('++++++++++ SVM ANALYSIS SECTION ++++++++++++++')

param_boxplots(abalone_svm_params, 
               plot_type = sns.swarmplot, 
               problem = 'abalone', 
               title = 'Abalone SVM Performance by Kernel Function',
               size = 15,)

param_boxplots(heart_svm_params, 
               plot_type = sns.swarmplot, 
               problem = 'heart', 
               size = 15,
               title = 'Heart SVM Performance by Kernel Function')

abalone_svm = SVC(random_state = 100, probability=True)
param_dist = {
    'kernel': ['rbf','poly'],
    'C':[0.001, 0.01, 1, 10, 100],
     'degree':[2,3,4,5,6]}

abalone_svm, cv_results = param_search(x_train_abalone,
                                       y_train_abalone,
                                       param_dist = param_dist,
                                       search_type = 'Grid',
                                       model = abalone_svm,
                                       verbose = True,
                                       scoring = 'roc_auc')

print('====================================================================')
print('PARAM SEARCH REPORT - SVM - ABALONE')
param_search_report(cv_results, n_top=100, save_as='analysis test svm', verbose=True, print_n_top=10)

heart_svm = SVC(random_state = 100, probability=True)
param_dist = {
    'kernel': ['rbf','poly'],
    'C':[0.001, 0.01, 1, 10, 100],
     'degree':[2,3,4,5,6]}

heart_svm, cv_results = param_search(x_train_heart,
                                       y_train_heart,
                                       param_dist = param_dist,
                                       search_type = 'Grid',
                                       model = abalone_svm,
                                       verbose = True,
                                       scoring = 'roc_auc')

print('====================================================================')
print('PARAM SEARCH REPORT - SVM - heart')
param_search_report(cv_results, n_top=100, save_as='analysis test svm', verbose=True, print_n_top=10)






print('++++++++++ Categorical Analysis ANALYSIS SECTION ++++++++++++++')


nominal_cols = ['resting_ecg', 'chest_pain_type','thal']
heart_dum = pd.get_dummies(heart, columns = nominal_cols, prefix = nominal_cols)

heart_cat = heart_dum.loc[:,['resting_ecg_0.0', 'resting_ecg_1.0', 'resting_ecg_2.0',
       'chest_pain_type_1.0', 'chest_pain_type_2.0', 'chest_pain_type_3.0',
       'chest_pain_type_4.0', 'thal_3.0', 'thal_6.0', 'thal_7.0','sex']]
heart_target = heart_dum.loc[:,'target']
heart_cont = heart_dum.loc[:,['age', 'resting_bp', 'serum_chol', 'fasting_bs', 'max_hr',
                   'angina', 'oldpeak', 'peak_st_slope', 'num_vessels']]

heart_cont = pd.DataFrame(scaler.fit_transform(heart_cont), columns = list(heart_cont.columns))
heart_dum = heart_cont.join(heart_cat)

heart_nn = pickle.load(open('heart_nn.p','rb'))
heart_dt = pickle.load(open('heart_dt.p','rb'))
heart_ada = pickle.load(open('heart_ada.p','rb'))
heart_knn = pickle.load(open('heart_knn.p','rb'))
heart_svm = pickle.load(open('heart_svm.p','rb'))


heart_post_categorical_analysis_df= pd.DataFrame({'Accuracy':[0,0,0,0,0], 'AUC':[0,0,0,0,0], 'APR':[0,0,0,0,0]}, index = ['DT','ANN','ADA','KNN','SVM'])

heart_nn.set_params(input_dim = 20)
heart_x = heart_dum.values

print('_______NN__________')
cv_results = cross_validate(heart_nn, heart_x, heart_target, cv = 3, scoring = ['accuracy','roc_auc','average_precision'])
heart_post_categorical_analysis_df.loc['ANN',:] = [np.average(cv_results['test_accuracy']), np.average(cv_results['test_roc_auc']), np.average(cv_results['test_average_precision'])]

print('_______DT__________')
cv_results = cross_validate(heart_dt, heart_x, heart_target, cv = 3, scoring = ['accuracy','roc_auc','average_precision'])
heart_post_categorical_analysis_df.loc['DT',:] = [np.average(cv_results['test_accuracy']), np.average(cv_results['test_roc_auc']), np.average(cv_results['test_average_precision'])]

print('________ada_________')
cv_results = cross_validate(heart_ada, heart_x, heart_target, cv = 3, scoring = ['accuracy','roc_auc','average_precision'])
heart_post_categorical_analysis_df.loc['ADA',:] = [np.average(cv_results['test_accuracy']), np.average(cv_results['test_roc_auc']), np.average(cv_results['test_average_precision'])]

print('_______knn________')
cv_results = cross_validate(heart_knn, heart_x, heart_target, cv = 3, scoring = ['accuracy','roc_auc','average_precision'])
heart_post_categorical_analysis_df.loc['KNN',:] = [np.average(cv_results['test_accuracy']), np.average(cv_results['test_roc_auc']), np.average(cv_results['test_average_precision'])]

print('________svm_________')
cv_results = cross_validate(heart_svm, heart_x, heart_target, cv = 3, scoring = ['accuracy','roc_auc','average_precision'])
heart_post_categorical_analysis_df.loc['SVM',:] = [np.average(cv_results['test_accuracy']), np.average(cv_results['test_roc_auc']), np.average(cv_results['test_average_precision'])]

heart_post_categorical_analysis_df.to_csv('post_cat_tuning_analysis_heart_problem.csv')
