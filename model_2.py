import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn import discriminant_analysis
import time

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_NAME_ = "Submissions/{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

_CROSS_VALIDATE_ = True

_NORMALIZE_ = True
_CREATE_ENSEMBLE_ = False

_RF_NUM_ESTIMATORS_ = 200
_ET_NUM_ESTIMATORS_ = 200
_GBM_ESTIMATORS_ = 1000

_ADABOOST_NUM_ESTIMATORS_ = 1000
_ADABOOST_LEARNING_RATE_ = 1
#_ADABOOST_BASE_ESTIMATOR_ = RandomForestClassifier(n_estimators=200,criterion="entropy",max_features=None,random_state=777,n_jobs=-1)
_ADABOOST_BASE_ESTIMATOR_ = ExtraTreesClassifier(n_estimators=10,random_state=777,n_jobs=-1)


_ONE_HOT_ENCODING_ = False
_CHECK_SIGN_ = False     #gbm does not give +1,-1 so tick this before running gbm

def read_data(file_name):
	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, field_types_file):

	# read filed_types file
	ft = open(field_types_file, "r")
	categ = []      # list of categorical variables for transform
	non_categ = []
	for line in ft.readlines():
		splits = line.split()
		if splits[1] != "numeric":
			categ.append(splits[0])

	print "Processing categorical feaatures", data.shape
	for c in categ:
		if _ONE_HOT_ENCODING_:
			data = pd.concat([data, pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))], axis=1)
		else:
			data[c] = preprocessing.LabelEncoder().fit_transform(data[c])
	print "Done processing categorical features", data.shape

	for column in data:
		if column not in categ:
			#data[column] = preprocessing.StandardScalar().fit_transform(data[column].reshape(-1,1))
			non_categ.append(column)

	if _NORMALIZE_:
		data[non_categ] = preprocessing.scale(data[non_categ])

	#imp = Imputer(missing_values='null', strategy='most_frequent', axis=0)
	#imp.fit(data)
	return data

def evaluate(predictions, labels):	
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples

def model_evaluate(model, test_data, test_labels):
	preds = model.predict(test_data)
	if _CHECK_SIGN_:
		preds[preds > 0]  = 1
		preds[preds <= 0] = -1
	return evaluate(preds.reshape(preds.size, 1), test_labels)

def lda(train_data, train_labels, test_data, test_labels):
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), lda

def random_forest(train_data, train_labels, test_data, test_labels):
	rfc = RandomForestClassifier(n_estimators=_RF_NUM_ESTIMATORS_)
	rfc.fit(train_data, train_labels.ravel())
	return model_evaluate(rfc, test_data, test_labels), rfc

def extra_tree_classifier(train_data, train_labels, test_data, test_labels):
	etc = ExtraTreesClassifier(n_estimators=_RF_NUM_ESTIMATORS_)
	etc.fit(train_data, train_labels.ravel())
	return model_evaluate(etc, test_data, test_labels), etc

def ada_boost_classifier(train_data, train_labels, test_data, test_labels):
	ada = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm="SAMME.R", learning_rate=_ADABOOST_LEARNING_RATE_)
	ada.fit(train_data, train_labels.ravel())
	return model_evaluate(ada, test_data, test_labels), ada

def gbm(train_data, train_labels, test_data, test_labels):
	params = {'n_estimators': _GBM_ESTIMATORS_, 'max_depth': 4, 'learning_rate': 0.5}
	clf = GradientBoostingClassifier(**params)

	clf.fit(train_data, train_labels.ravel())
	#mse = mean_squared_error(train_labels, clf.predict(train_data))
	#print("MSE: %.4f" % mse)
	return model_evaluate(clf, test_data, test_labels), clf

def logistic_regression_mle(train_data, train_labels, test_data, test_labels):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), logreg

def qda(train_data, train_labels, test_data, test_labels):
	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
	qda.fit(train_data, train_labels.ravel())
	return model_evaluate(qda, test_data, test_labels), qda

def create_ensemble(train_data, train_labels):
	ensemble = []

	train_data_1, train_data_2, train_labels_1, train_labels_2 = \
						cross_validation.train_test_split(train_data, train_labels, test_size=0.75, random_state=0)

	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(train_data_1, train_labels_1.ravel())

	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(train_data_1, train_labels_1.ravel())

	qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
	qda.fit(train_data_1, train_labels_1.ravel())

	preds_lda = lda.predict(train_data_2)
	preds_logreg = logreg.predict(train_data_2)
	preds_qda = qda.predict(train_data_2)

	# train_data_2["lda"] = preds_lda
	# train_data_2["logreg"] = preds_logreg
	# train_data_2["qda"] = preds_qda

	train_data_2 = pd.concat([pd.DataFrame(preds_lda), pd.DataFrame(preds_logreg), pd.DataFrame(preds_qda)], axis='1')
	ensemble = [lda, logreg, qda]

	return train_data_2, train_labels_2, ensemble

def append_test_data(test_data, ensemble):
	preds_lda = ensemble[0].predict(test_data)
	preds_logreg = ensemble[1].predict(test_data)
	preds_qda = ensemble[2].predict(test_data)

	# test_data["lda"] = preds_lda
	# test_data["logreg"] = preds_logreg
	# test_data["qda"] = preds_qda

	test_data = pd.concat([pd.DataFrame(preds_lda), pd.DataFrame(preds_logreg), pd.DataFrame(preds_qda)], axis='1')
	return test_data


def write_preds_to_file(file_name, df):
	df.to_csv(file_name, header=_OUTPUT_FILE_HEADER_, index_label="Id")

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data = preprocess_data(train_data, _FIELDS_FILE_)

	if _CREATE_ENSEMBLE_:
		train_data, train_labels, ensemble = create_ensemble(train_data, train_labels)

	err, model = ada_boost_classifier(train_data, train_labels, train_data, train_labels)
	print "###############################################"
	print "Trianing error rate:", err
	print "###############################################"

	# cross-validate
	if _CROSS_VALIDATE_:
		#nrfc = ExtraTreesClassifier(n_estimators=_ET_NUM_ESTIMATORS_, random_state=42)
		nrfc = AdaBoostClassifier(_ADABOOST_BASE_ESTIMATOR_, n_estimators=_ADABOOST_NUM_ESTIMATORS_, algorithm="SAMME.R", learning_rate=_ADABOOST_LEARNING_RATE_)
		kfold = cross_validation.KFold(len(train_data), n_folds=5)
		scores = cross_validation.cross_val_score(nrfc, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
		print "Cross-validation score:", sum(scores)/len(scores)
		print "###############################################"
	
	test_data = read_data(_TEST_FILE_NAME_)
	test_data = preprocess_data(test_data, _FIELDS_FILE_)

	if _CREATE_ENSEMBLE_:
		test_data = append_test_data(test_data, ensemble)

	preds = model.predict(test_data)
	preds_df = pd.DataFrame(preds)
	preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
	write_preds_to_file(output_fname, preds_df)


