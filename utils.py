import numpy as np
import pandas as pd
from IPython.core.display import display
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score, cross_validate
from keras.models import Sequential
import seaborn as sns
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
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def gridsearch_heatmap(arr, xlab, ylab, title, save_as = None, color_bar = True, xtick_marks = False, rotation = 0):
    '''currently only works with two search parameters'''
    plt.figure(figsize = (8,6))
    plt.subplots_adjust(left = 0.2, right = 0.95, bottom = 0.15, top = 0.95)
    plt.imshow(arr, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if color_bar:
        plt.colorbar().set_label('AUC')
    plt.xticks(np.arange(arr.shape[1]), labels = list(arr.columns), rotation = rotation)
    plt.yticks(np.arange(arr.shape[0]), labels = list(arr.index))
    plt.title(title)
    if save_as:
        plt.savefig(save_as + '.png')
    plt.show()
    
def param_boxplots(df, problem, title, plot_type = sns.boxplot,log = False, size = None):
    
    cols = list(df.columns)
    drop_cols = ['Unnamed: 0','test_score','rank','test_stds','fit_time','score_time']
    for x in drop_cols:
        cols.remove(x)
        
    for param in cols:
        if size:
            plot_type(x = param, 
                        y = 'test_score',
                        data = df,
                        palette = 'muted',
                        size = size)
            if log:
                plt.yscale('log')
            plt.ylabel('AUC')
            plt.title(title)
            plt.savefig(problem + param + '.png')
            plt.show()
        else:
            plot_type(x = param, 
                        y = 'test_score',
                        data = df,
                        palette = 'muted')
            if log:
                plt.yscale('log')
            plt.ylabel('AUC')
            plt.title(title)
            plt.savefig(problem + param + '.png')
            plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), save_as = None, scoring = None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    if save_as:
        plt.savefig(save_as + '.png')
    
    return plt
        
def param_search_report(results, n_top, save_as, print_n_top=3, verbose = False):
    if verbose:
        for i in range(1, print_n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    params = list(results['params'][0].keys())
    values = []
    ranks = []
    test_scores = []
    test_stds = []
    fit_times = []
    score_times = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            ranks.append(i)
            fit_times.append(results['mean_fit_time'][candidate])
            score_times.append(results['mean_score_time'][candidate])
            test_scores.append(results['mean_test_score'][candidate])
            test_stds.append(results['std_test_score'][candidate])
            values.append(list(results['params'][candidate].values()))
    full_param_report = pd.DataFrame(data = values, columns = params)
    full_param_report['test_score'] = test_scores
    full_param_report['rank'] = ranks
    full_param_report['test_stds'] = test_stds
    full_param_report['fit_time'] = fit_times
    full_param_report['score_time'] = score_times
    full_param_report.to_csv(save_as + '.csv')

def create_nn_model(input_dim, layer1_nodes = 20, layer2_nodes = 20, learn_rate = 0.01, momentum = 0.9,loss = 'mean_squared_error', layer1_initializer = 'uniform', layer2_initializer = 'uniform', add_dropout = False, dropout_param = None):
    ## build model
    model = Sequential()
    if add_dropout:
        model.add(Dropout(dropout_param, input_shape=(input_dim,)))
    model.add(Dense(layer1_nodes, input_dim=input_dim, kernel_initializer=layer1_initializer, activation='relu'))
    model.add(Dense(layer2_nodes, kernel_initializer=layer2_initializer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    ## compile 'model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss=loss, optimizer='SGD', metrics=['accuracy'])
    return model

def param_search(x, y, param_dist, model, search_type, iterations = 20, verbose = False, scoring = None):
    '''
    search_type: 'Grid','Random'
    '''
    ## initialize random search for good parameters
    if search_type=='Random':
        search_results = RandomizedSearchCV(model, param_dist, n_iter=iterations, n_jobs=-1, cv=3,
                                            scoring=scoring)
        search_results.fit(x,y)
    if search_type=='Grid':
        search_results = GridSearchCV(model, param_dist, n_jobs=-1, cv=3,
                                      scoring=scoring)
        search_results.fit(x,y)
    return search_results.best_estimator_, search_results.cv_results_

def clf_metrics_report(x_test, y_test, model, verbose=False):
    y_pred = model.predict(x_test)
    proba_pos_class = model.predict_proba(x_test)
    proba_pos_class[:,1]
    ps = precision_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    acc = model.score(x_test, y_test)
    roc = roc_auc_score(y_test, proba_pos_class)
    if verbose:
        print("precision score:", ps)
        print("recall score:", rs)
        print("accuracy score:", acc)
    return ps, rs, acc, roc