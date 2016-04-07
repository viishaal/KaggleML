import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn import discriminant_analysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time

#### MODEL CONFIG
_RF_NUM_ESTIMATORS_ = 250
_ET_NUM_ESTIMATORS_ = 400
_GBM_ESTIMATORS_ = 1

_ADABOOST_NUM_ESTIMATORS_ = 50
_ADABOOST_LALGO_ = "SAMME"
_ADABOOST_LEARNING_RATE_ = 1
#_ADABOOST_BASE_ESTIMATOR_ = RandomForestClassifier(n_estimators=200,criterion="entropy",max_features=None,random_state=777,n_jobs=-1)
_ADABOOST_BASE_ESTIMATOR_ = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_,random_state=777,n_jobs=-1)

_CHECK_SIGN_ = False     #gbm does not give +1,-1 so tick this before running gbm



def evaluate(predictions, labels):	
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples

def model_evaluate(model, test_data, test_labels):
	preds = model.predict(test_data)
	if _CHECK_SIGN_:
		preds[preds > 0]  = 1
		preds[preds <= 0] = -1
	return evaluate(preds.reshape(preds.size, 1), test_labels)


def cross_validate_model(train_data, train_labels, kfolds, model):
	#nrfc = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_, random_state=42)
	#nrfc = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm=_ADABOOST_LALGO_, learning_rate=_ADABOOST_LEARNING_RATE_)
	kfold = cross_validation.KFold(len(train_data), n_folds=kfolds)
	scores = cross_validation.cross_val_score(model, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
	print "Cross-validation score:", sum(scores)/len(scores)
	print "###############################################"

def lda(train_data, train_labels, test_data, test_labels):
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), lda

def random_forest(train_data, train_labels, test_data, test_labels):
	rfc = RandomForestClassifier(n_estimators=_RF_NUM_ESTIMATORS_)
	rfc.fit(train_data, train_labels.ravel())
	return model_evaluate(rfc, test_data, test_labels), rfc

def extra_tree_classifier(train_data, train_labels, test_data, test_labels):
	etc = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_)
	etc.fit(train_data, train_labels.ravel())
	return model_evaluate(etc, test_data, test_labels), etc

def ada_boost_classifier(train_data, train_labels, test_data, test_labels):
	ada = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm=_ADABOOST_LALGO_, learning_rate=_ADABOOST_LEARNING_RATE_)
	ada.fit(train_data, train_labels.ravel())
	return model_evaluate(ada, test_data, test_labels), ada

def gbm(train_data, train_labels, test_data, test_labels):
	params = {'n_estimators': _GBM_ESTIMATORS_, 'max_depth': 4, 'learning_rate': 0.5}
	clf = GradientBoostingClassifier(**params)

	clf.fit(train_data, train_labels.ravel())
	#mse = mean_squared_error(train_labels, clf.predict(train_data))
	#print("MSE: %.4f" % mse)
	return model_evaluate(clf, test_data, test_labels), clf

def gbr(train_data, train_labels, test_data, test_labels):
	est = GradientBoostingRegressor(n_estimators=_GBM_ESTIMATORS_, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
	est.fit(train_data, train_labels.ravel())
	return model_evaluate(est, test_data, test_labels), est

def logistic_regression_mle(train_data, train_labels, test_data, test_labels):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), logreg

def qda(train_data, train_labels, test_data, test_labels):
	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
	qda.fit(train_data, train_labels.ravel())
	return model_evaluate(qda, test_data, test_labels), qda

def svm(train_data, train_labels, test_data, test_labels):
	svm = SVC();
	svm.fit(train_data, train_labels.ravel())
	return model_evaluate(svm, test_data, test_labels), svm

def nn(train_data, train_labels, test_data, test_labels):
	clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(200,), verbose=False, random_state=55)
	clf.fit(train_data, train_labels.ravel())
	return model_evaluate(clf, test_data, test_labels), clf


### add model function to this dictionary once you define it above
_ESTIMATORS_META_ = {
								"gbm": gbm,
								"adaboost": ada_boost_classifier, 
								"etc": extra_tree_classifier,
								"rf": random_forest,
								"neural": nn,
								"lda": lda,
								"qda": qda,
								"gbr": gbr
					}


## ENSEMBLER CONFIG
_ENSEMBLE_STACK_ = [
						"lda",
						"qda",
						"gbm",
						"gbr",
						"rf",
					]


def create_ensemble(train_data, train_labels):
	ensemble = []

	train_data_1, train_data_2, train_labels_1, train_labels_2 = \
						cross_validation.train_test_split(train_data, train_labels, test_size=0.75, random_state=0)

	ensemble = []
	for m in _ENSEMBLE_STACK_:
		err, model = _ESTIMATORS_META_[m](train_data_1, train_labels_1, train_data_1, train_labels_1)
		print "training error rate of ", m, " is ", err 
		ensemble.append(model)

	predictions = []
	for i, model in enumerate(ensemble):
		predictions.append(model.predict(train_data_2))

	for i, prediction in enumerate(predictions):
		train_data_2["nf_"+str(i)] = prediction

	return train_data_2, train_labels_2, ensemble


def append_test_data(test_data, ensemble):
	predictions = []
	for i, model in enumerate(ensemble):
		predictions.append(model.predict(test_data))

	for i, prediction in enumerate(predictions):
		test_data["nf_"+str(i)] = prediction

	#test_data = pd.concat([pd.DataFrame(preds_lda), pd.DataFrame(preds_logreg), pd.DataFrame(preds_qda), pd.DataFrame(preds_svm), pd.DataFrame(preds_rf)], axis='1')
	return test_data

