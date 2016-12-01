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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time

#### MODEL CONFIG
_RF_NUM_ESTIMATORS_ = 400
_ET_NUM_ESTIMATORS_ = 300
_GBM_ESTIMATORS_ = 1000

_ADABOOST_NUM_ESTIMATORS_ = 100
_ADABOOST_LALGO_ = "SAMME"
_ADABOOST_LEARNING_RATE_ = 1
#_ADABOOST_BASE_ESTIMATOR_ = RandomForestClassifier(n_estimators=200,criterion="entropy",max_features=None,random_state=777,n_jobs=-1)
_ADABOOST_BASE_ESTIMATOR_ = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_,random_state=777,n_jobs=-1)

_CHECK_SIGN_ = False     #gbm does not give +1,-1 so tick this before running gbm



def evaluate(predictions, labels):
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples

def model_evaluate(model, test_data, test_labels):
	"""
	Evaluates accuracy of predicted model labels by comparing with test labels

	Args:
		model: fitted model
		test_data: test_data from which predicted labels are generated
		test_labels: true labels of the given dataset

	Returns:
		accuracy: accuracy of the model as a fraction

	"""

	preds = model.predict(test_data)
	if _CHECK_SIGN_:
		preds[preds > 0]  = 1
		preds[preds <= 0] = -1
	return evaluate(preds.reshape(preds.size, 1), test_labels)

def fit_model(model, train_data, train_labels, test_data=None, test_labels=None):
	model.fit(train_data, train_labels.ravel())
	if test_data is None or test_labels is None:
		return model_evaluate(model, train_data, train_labels)
	else:
		return model_evaluate(model, test_data, test_labels)

def cross_validate_model(train_data, train_labels, kfolds, model):
	"""
	Performs kfold cross-validation for the provided model
	Prints accuracy on screen

	Args:
		train_data: training data
		train_labels: true labels for training dataset
		kfolds: number of folds
		model: scikit trained model

	Returns:
		None

	"""

	#nrfc = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_, random_state=42)
	#nrfc = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm=_ADABOOST_LALGO_, learning_rate=_ADABOOST_LEARNING_RATE_)
	kfold = cross_validation.KFold(len(train_data), n_folds=kfolds)
	scores = cross_validation.cross_val_score(model, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
	print "###############################################"
	print "Cross-validation score:", sum(scores)/len(scores)
	print "###############################################"

### Use below functions for model instantiation

# the list of models used for this task 
# the paramteres for different models are defined on top of this file or inside the functions

# Following set of models are available:
# 1) LDA
# 2) Random Forests
# 3) Extra Tree Classifier
# 4) Ada Boost
# 5) Gradient Boosting Trees
# 6) Gradient Boosting Regressor
# 7) Logistic Regression
# 8) Quadratic Discriminant Analysis
# 9) Support Vector Machine


def lda():
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	return lda

def random_forest():
	rfc = RandomForestClassifier(n_estimators=_RF_NUM_ESTIMATORS_)
	return rfc

def extra_tree_classifier():
	etc = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_)#, max_depth=30)#, max_features=None)
	return etc

def ada_boost_classifier():
	ada = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm=_ADABOOST_LALGO_, learning_rate=_ADABOOST_LEARNING_RATE_)
	return ada

def gbm():
	params = {'n_estimators': _GBM_ESTIMATORS_, 'max_depth': 4, 'learning_rate': 0.5}
	clf = GradientBoostingClassifier(**params)
	return gbm

def gbr():
	est = GradientBoostingRegressor(n_estimators=_GBM_ESTIMATORS_, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
	return est

def logistic_regression_mle():
	logreg = linear_model.LogisticRegression()
	return logreg

def qda():
	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
	return qda

def svm():
	svm = SVC();
	return svm


def knn(n_neighbors=5):
	knn = KNeighborsClassifier(n_neighbors, weights="uniform", n_jobs=-1)
	return knn

### add model function to this dictionary once you define it above
_ESTIMATORS_META_ = {
								"gbm": gbm,
								"adaboost": ada_boost_classifier, 
								"etc": extra_tree_classifier,
								"rf": random_forest,
								"logreg": logistic_regression_mle,
								"lda": lda,
								"qda": qda,
								"gbr": gbr,
								"knn": knn
					}


## ENSEMBLER CONFIG
_ENSEMBLE_STACK_ = [
						"lda",
						"qda",
						"gbm",
						#"gbr",
						"rf",
					]


def create_ensemble(train_data, train_labels):
	"""
	Creates an ensemble of model using the _ENSEMBLE_STACK_ defined above
	_ENSEMBLE_STACK_ defines the stack of models used in the ensemble

	The ensemble is created by splitting the training data into 2 parts
	part 1 is used to train the ensemble models
	predictions from each of the ensemble models are appended as predictions to
	part 2 of the datatset with prediction probabilities of the indiviudal classifiers

	the part 2 of the dataset is used to train another model finally

	Args:
		train_data: training data
		train_labels: training labels

	Returns:
		train_data_2: enhanced training data with ensemble predictions and confidence scores
		train_labels_2: labels for training data
		ensemble: trained ensemble of models

	"""

	ensemble = []

	train_data_1, train_data_2, train_labels_1, train_labels_2 = \
						cross_validation.train_test_split(train_data, train_labels, test_size=0.75, random_state=0)

	ensemble = []
	for m in _ENSEMBLE_STACK_:
		model = _ESTIMATORS_META_[m]()
		err = fit_model(model, train_data_1, train_labels_1, None, None)
		print "training error rate of ", m, " is ", err 
		ensemble.append(model)

	predictions = []
	for i, model in enumerate(ensemble):
		predictions.append(model.predict_proba(train_data_2)[:,1])

	for i, prediction in enumerate(predictions):
		train_data_2["nf_"+str(i)] = prediction

	return train_data_2, train_labels_2, ensemble


def append_test_data(test_data, ensemble):
	"""
		Generate predictions and confidence scores for test data
	"""

	predictions = []
	for i, model in enumerate(ensemble):
		predictions.append(model.predict_proba(test_data)[:,1])

	for i, prediction in enumerate(predictions):
		test_data["nf_"+str(i)] = prediction

	#test_data = pd.concat([pd.DataFrame(preds_lda), pd.DataFrame(preds_logreg), pd.DataFrame(preds_qda), pd.DataFrame(preds_svm), pd.DataFrame(preds_rf)], axis='1')
	return test_data

def blend_models(n_folds, train_data, train_labels, holdout, test_data, test_mode):
	"""
	Function which performs the blending procedure explained below:

	Step 1) initialize classifiers to use in the blending task as clfs variable (add classifiers to that variable)
	TODO: extract it out into config

	Step 2) split training data into kfolds

	Step 3) 
	Do for every classifier:
		Do for every fold:
			train each classifier in blender on the kth training fold and do the following:
				a) predict probabilities of the kth "test" fold only
				b) append predictions to holdout set "dataset_blend_holdout_j" for classifier trained on that fold only
				c) append predictions to test set "dataset_blend_test_j" for classifier trained on that fold only

		When all folds are finished processing take a mean of predictions generated for the classifier trained on different folds
		for both dataset_blend_holdout_j and dataset_blend_test_j and append mean values to dataset_blend_holdout, dataset_blend_test

	Args:
		n_folds: number of folds in the blender
		train_data: trianing data
		train_labels: true labels for training data
		holdout: holdout set
		test_data: test data set to generate final predictions on
		test_mode: this is the debug mode (it uses only one classifier in the blender)

	Returns:
		dataset_blend_train: blended training data set based on above procedure
		dataset_blend_holdout: blended holdout set based on above procedure
		dataset_blend_test: blended test set based on above procedure

	"""

	np.random.seed(0) # seed to shuffle the train set

	shuffle = False
	X = train_data
	y = train_labels.ravel()
	X_submission = holdout

	if shuffle:
		idx = np.random.permutation(y.size)
		X = X[idx]
		y = y[idx]
	skf = list(cross_validation.KFold(len(y), n_folds))

	if test_mode:
		clfs = [KNeighborsClassifier(weights="uniform", n_jobs=-1)]
	else:
		clfs = [KNeighborsClassifier(weights="uniform", n_jobs=-1),
			KNeighborsClassifier(weights="distance", n_jobs=-1),
			SVC(),
			RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion='gini'),
			RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion='entropy'),
			ExtraTreesClassifier(n_estimators=250, n_jobs=-1, criterion='gini'),
			ExtraTreesClassifier(n_estimators=250, n_jobs=-1, criterion='entropy'),
			GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
			discriminant_analysis.LinearDiscriminantAnalysis(),
			discriminant_analysis.QuadraticDiscriminantAnalysis()]
			#MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(200,), verbose=False, random_state=55),
			#AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm=_ADABOOST_LALGO_, learning_rate=_ADABOOST_LEARNING_RATE_)]

	print "Creating train and test sets for blending."

	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_holdout = np.zeros((X_submission.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((test_data.shape[0], len(clfs)))

	for j, clf in enumerate(clfs):
		print "Classifier no: ", j + 1
		print clf
		dataset_blend_holdout_j = np.zeros((X_submission.shape[0], len(skf)))
		dataset_blend_test_j = np.zeros((test_data.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print "====Fold", i
			X_train = X.iloc[train]
			y_train = y[train]
			X_test = X.iloc[test]
			y_test = y[test]
			clf.fit(X_train, y_train)
			y_submission = clf.predict_proba(X_test)[:,1]
			dataset_blend_train[test, j] = y_submission
			dataset_blend_holdout_j[:, i] = clf.predict_proba(X_submission)[:,1]
			dataset_blend_test_j[:, i] = clf.predict_proba(test_data)[:,1]
		dataset_blend_holdout[:,j] = dataset_blend_holdout_j.mean(1)
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	return pd.DataFrame(dataset_blend_train), pd.DataFrame(dataset_blend_holdout), pd.DataFrame(dataset_blend_test)



