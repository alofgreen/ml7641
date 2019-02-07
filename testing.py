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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
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

from utils import param_search_report, create_nn_model, param_search, plot_learning_curve

random_state = 100

%matplotlib inline

##load decision tree
abalone_dt = pickle.load(open('abalone_dt.p', 'rb'))
heart_dt = pickle.load(open('heart_dt.p','rb'))
##load nn
abalone_nn = pickle.load(open('abalone_nn.p','rb'))
heart_nn = pickle.load(open('heart_nn.p','rb'))
##load ada boost
abalone_ada = pickle.load(open('abalone_ada.p','rb'))
heart_ada = pickle.load(open('heart_ada.p','rb'))
##load knn
abalone_knn = pickle.load(open('abalone_knn.p','rb'))
heart_knn = pickle.load(open('heart_knn.p','rb'))
##load svm
abalone_svm = pickle.load(open('abalone_svm.p','rb'))
abalone_svm.set_params(probability=True)
heart_svm = pickle.load(open('heart_svm.p','rb'))
heart_svm.set_params(probability=True)


##print model parameters
print('+++++++++PRINT MODEL PARAMETERS++++++++++++')
print('abalone dt params')
print(abalone_dt.get_params())
print('===========================================')
print('')
print('heart dt params')
print(heart_dt.get_params())
print('===========================================')
print('')
print('abalone nn params')
print(abalone_nn.get_params())
print('===========================================')
print('')
print('heart nn params')
print(heart_nn.get_params())
print('===========================================')
print('')
print('abalone ada params')
print(abalone_ada.get_params())
print('===========================================')
print('')
print('heart ada params')
print(heart_ada.get_params())
print('===========================================')
print('')
print('abalone knn params')
print(abalone_knn.get_params())
print('===========================================')
print('')
print('heart knn params')
print(heart_knn.get_params())
print('===========================================')
print('')
print('abalone svm params')
print(abalone_svm.get_params())
print('===========================================')
print('')
print('heart svm params')
print(heart_svm.get_params())

abalone_models = [abalone_dt, abalone_nn, abalone_ada, abalone_knn, abalone_svm]
heart_models = [heart_dt, heart_nn, heart_ada, heart_knn, heart_svm]
dt_models = [abalone_dt, heart_dt]
nn_models = [abalone_nn, heart_nn]
ada_models = [abalone_ada, heart_ada]
knn_models = [abalone_knn, heart_knn]
svm_models = [abalone_svm, heart_svm]
model_types = [dt_models, nn_models, ada_models, knn_models, svm_models]

## DATASET PREP
heart_colnames = ['age','sex','chest_pain_type',
                  'resting_bp','serum_chol','fasting_bs',
                  'resting_ecg','max_hr','angina','oldpeak',
                  'peak_st_slope','num_vessels','thal','heart_disease']
abalone_colnames = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']

## LOAD DATA
heart = pd.read_table('/Users/Andy/Documents/OMSCS/machine_learning/heart.txt', sep=' ', header=None, names=heart_colnames)
# spambase = pd.read_csv('/Users/Andy/Documents/OMSCS/machine_learning/spambase.txt',header=None)
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
heart_y = heart.iloc[:,-1].values

## ABALONE - SEP. TARGET AND FEATURES
abalone_x = abalone.drop('target',axis=1).values
scaler = StandardScaler()
abalone_x = scaler.fit_transform(abalone_x)
abalone_y = abalone['target'].values
data_all = [(abalone_x, abalone_y), (heart_x, heart_y)]

plot_learning_curve(estimator=abalone_dt, 
                    title='DT Learning Curve - Abalone', 
                    X = abalone_x, 
                    y = abalone_y, 
                    cv = 3, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    n_jobs = -1,
                    save_as = 'abalone_dt_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = heart_dt, 
                    title = 'DT Learning Curve - Heart',
                    X = heart_x, 
                    y = heart_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1, 
                    save_as = 'heart_dt_lc',
                    scoring = 'roc_auc'
                   )

plot_learning_curve(estimator = abalone_nn, 
                    title = 'NN Learning Curve - Abalone',
                    X = abalone_x, 
                    y = abalone_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs=-1, 
                    save_as = 'abalone_nn_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator= heart_nn, 
                    title = 'NN Learning Curve - Heart',
                    X = heart_x, 
                    y = heart_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1, 
                    save_as = 'heart_nn_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = abalone_ada, 
                    title = 'Ada Boost Learning Curve - Abalone',
                    X = abalone_x, 
                    y = abalone_y,
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv =3,
                    n_jobs = -1, 
                    save_as = 'abalone_ada_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator= heart_ada, 
                    title = 'Ada Boost Learning Curve - Heart',
                    X = heart_x,
                    y = heart_y, 
                    cv = 3, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    n_jobs = -1, 
                    save_as = 'heart_ada_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = abalone_knn, 
                    title = 'KNN Learning Curve - Abalone',
                    X = abalone_x, 
                    y = abalone_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1,
                    save_as = 'abalone_knn_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = heart_knn, 
                    title = 'KNN Learning Curve - Heart',
                    X = heart_x, 
                    y = heart_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1, 
                    save_as = 'heart_knn_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = abalone_svm, 
                    title = 'SVM Learning Curve - Abalone', 
                    X = abalone_x, 
                    y = abalone_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1,
                    save_as = 'abalone_svm_lc',
                    scoring = 'roc_auc')

plot_learning_curve(estimator = heart_svm, 
                    title = 'SVM Learning Curve - Heart',
                    X = heart_x, 
                    y = heart_y, 
                    train_sizes = np.linspace(0.15, 1.0, 5),
                    cv = 3, 
                    n_jobs = -1, 
                    save_as = 'heart_svm_lc',
                    scoring = 'roc_auc')


print('''For  each model within each problem type analyze the accuracy, area under the ROC curve (AUC) and average precision
score (APR).  Create a table with the scores averaged accross each problem and another with all scores for each problem.
''')
avg_scores_list = []
all_scores_list = []
avg_times_list = []
for problem in model_types:
    train_acc_list = []
    test_acc_list = []
    train_auc_list = []
    test_auc_list = []
    train_apr_list = []
    test_apr_list = []
    fit_time_list = []
    score_time_list = []
    for model, dataset in zip(problem, data_all):
        ##datasets
        X = dataset[0]
        y = dataset[1]
        cv_results = cross_validate(model, X, y, cv = 3, scoring = ('average_precision', 'roc_auc', 'accuracy'))
        
        fit_time = np.average(cv_results['fit_time'])
        score_time = np.average(cv_results['score_time'])
        apr = np.average(cv_results['test_average_precision'])
        auc = np.average(cv_results['test_roc_auc'])
        acc = np.average(cv_results['test_accuracy'])
        
        apr_train = np.average(cv_results['train_average_precision'])
        auc_train = np.average(cv_results['train_roc_auc'])
        acc_train = np.average(cv_results['train_accuracy'])
        
        #test scores
        test_auc_list.append(auc)
        test_apr_list.append(apr)
        test_acc_list.append(acc)
        
        #train scores
        train_auc_list.append(auc_train)
        train_apr_list.append(apr_train)
        train_acc_list.append(acc_train)
        
        #time
        score_time_list.append(score_time)
        fit_time_list.append(fit_time)
                
        all_scores_list.append([acc, acc_train, auc,  auc_train, apr, apr_train])
    avg_scores_list.append([np.average(test_acc_list),
                            np.average(train_acc_list),
                            np.average(test_auc_list),
                            np.average(train_auc_list),
                            np.average(test_apr_list),
                            np.average(train_apr_list)])
    avg_times_list.append([np.average(fit_time_list), np.average(score_time_list)])
                           
df_index = ['DT', 'ANN', 'ADA', 'KNN', 'SVM']
df_cols = ['Test Acc.', 'Train Acc', 'Test AUC', 'Train AUC','Test APR', 'Train APR']

avg_scores_df = pd.DataFrame(data = np.array(avg_scores_list), 
                         index = df_index, 
                         columns = df_cols)
avg_scores_df.to_csv('avg_scores.csv')

heart_scores = all_scores_list[1::2]
abalone_scores = all_scores_list[::2]
abalone_scores_df = pd.DataFrame(abalone_scores, columns = df_cols, index = df_index)
heart_scores_df = pd.DataFrame(heart_scores, columns = df_cols, index = df_index)
all_scores_df = abalone_scores_df.join(heart_scores_df, lsuffix = '_abalone', rsuffix = '_heart')
all_scores_df.to_csv('all_model_scores.csv')

avg_times_df = pd.DataFrame(data = np.array(avg_times_list),
                            index = df_index,
                            columns = ['Fit Time','Score Time'])
avg_times_df.to_csv('avg_times.csv')

## Choose 0.775 for test train split
## HEART - TRAIN TEST SPLIT
x_train_heart, x_test_heart, y_train_heart, y_test_heart = \
    train_test_split(heart_x, heart_y, test_size = 0.225, random_state = random_state)
heart_data = (x_train_heart, x_test_heart, y_train_heart, y_test_heart)
## ABALONE - TRAIN TEST SPLIT
x_train_abalone, x_test_abalone, y_train_abalone, y_test_abalone = \
    train_test_split(abalone_x, abalone_y, test_size = 0.225, random_state = random_state)
abalone_data = (x_train_abalone, x_test_abalone, y_train_abalone, y_test_abalone)

abalone_nn.fit(x_train_abalone, y_train_abalone, validation_split = 0.3)
heart_nn.fit(x_train_heart, y_train_heart, validation_split = 0.3)

## Create learning curve plots for abalone neural network
## Create learning curve plots for abalone
plt.figure(figsize=(10,4))
plt.plot((abalone_nn.model.history.history['loss']))
plt.plot((abalone_nn.model.history.history['val_loss']))
plt.legend(['Train','Validation'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('abalone_nn_epoch_lc.png')
plt.show()

## create learning curve plots for heart neural network
plt.figure(figsize=(10,4))
plt.plot((heart_nn.model.history.history['loss']))
plt.plot((heart_nn.model.history.history['val_loss']))
plt.legend(['Train','Validation'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('heart_nn_epoch_lc.png')
plt.show()

real_test_errors = []
heart_ada = abalone_ada.fit(x_train_abalone, y_train_abalone)
for real_test_predict in abalone_ada.staged_predict(x_test_abalone):
    real_test_errors.append(1. - recall_score(real_test_predict, y_test_abalone))
n_trees_discrete = len(abalone_ada)
plt.figure()
plt.plot(range(1, n_trees_discrete + 1),
         real_test_errors, c='blue', label='SAMME')
plt.legend()
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.title('Test Error With Number of Trees')
plt.savefig('abalone ada boosted tree learning curve')

real_test_errors = []
heart_ada = heart_ada.fit(x_train_heart, y_train_heart)
for real_test_predict in heart_ada.staged_predict(x_test_heart):
    real_test_errors.append(1. - recall_score(real_test_predict, y_test_heart))
n_trees_discrete = len(heart_ada)
plt.figure()
plt.plot(range(1, n_trees_discrete + 1),
         real_test_errors, c='blue', label='SAMME')
plt.legend()
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.title('Test Error With Number of Trees')
plt.savefig('heart ada boosted tree learning curve.png')
