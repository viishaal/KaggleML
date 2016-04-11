from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from read_data import *

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 17:25:55 2016

@author: nitishr314
"""

"""Kaggle competition: Predicting a Biological Response.
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. 


"""

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))

def evaluate(predictions, labels):	
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples



_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_HEADER_ = ["Prediction"]


_OUTPUT_FILE_NAME_ = "{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

_CROSS_VALIDATE_ = True

_NORMALIZE_ = True
_CREATE_ENSEMBLE_ = True

_ONE_HOT_ENCODING_ = False
_K_FOLDS_ = 5

_MAIN_ESTIMATOR_ = "gbm"

_SPLIT_CATEGORICAL_ = False


if __name__ == '__main__':
    
    data = read_data(_TRAINING_FILE_NAME_)
    #print data.describe()
    train_labels = data.label
    train_labels = train_labels.reshape(train_labels.size, 1)
    train_data = data.drop("label", 1)

    train_data, les, lbs = preprocess_data(train_data, _FIELDS_FILE_,_NORMALIZE_, _ONE_HOT_ENCODING_, None, None, _SPLIT_CATEGORICAL_)
    train_data_2, train_data_1, train_labels_2, train_labels_1 = \
						cross_validation.train_test_split(train_data, train_labels, test_size=0.75, random_state=0)


    train_data_2 =train_data_2.drop(['18','20','23','25','26','58'], axis=1)
    train_data_1 =train_data_1.drop(['18','20','23','25','26','58'], axis=1)







    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False
    X = train_data_1
    y = train_labels_1.ravel()
    #y = train_labels_1
    X_submission = train_data_2

  #  X, y, X_submission = load_data.load()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    skf = list(KFold(len(y),n_folds))
#    skf = list(KFold(len(y),n_folds))
    #skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=250, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=250, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X.iloc[train]
            y_train = y[train]
            X_test = X.iloc[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]
    
    
    for index,j in enumerate(y_submission):
        if y_submission[index] > 0.5:
            y_submission[index] = 1
        else:
            y_submission[index] = -1
    a=0;
    for index,j in enumerate(y_submission):
        if y_submission[index] == train_labels_2[index]:
            print 'opo'
        else:    
            a = a+1;
    err = 1 - (a/train_labels_2.size);   
    print err
    
    

    #evaluate(y_submission, train_labels_2)    
    
    preds_df = pd.DataFrame(y_submission)
    preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']
    output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
    write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)
    

