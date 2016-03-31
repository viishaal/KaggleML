import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import time

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_NAME_ = "Submissions/{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

def read_data(file_name):
	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, field_types_file, replace_missing):

	# read filed_types file
	ft = open(field_types_file, "r")
	categ = []      # list of categorical variables for transform
	non_categ = []
	for line in ft.readlines():
		splits = line.split()
		if splits[1] != "numeric":
			categ.append(splits[0])

	# data preprocessing
	if (replace_missing):
		data = data.replace(["null"], [None])
		data = data.dropna(how='any')

	for c in categ:
		data[c] = preprocessing.LabelEncoder().fit_transform(data[c])

	for column in data:
		if column not in categ:
			data[column] = preprocessing.MinMaxScaler().fit_transform(data[column].reshape(-1,1))

	#imp = Imputer(missing_values='null', strategy='most_frequent', axis=0)
	#imp.fit(data)
	return data

def evaluate(predictions, labels):	
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples

def model_evaluate(model, test_data, test_labels):
	preds = model.predict(test_data)
	return evaluate(preds.reshape(preds.size, 1), test_labels)

def lda(train_data, train_labels, test_data, test_labels):
	lda = discriminant_analysis.LinearDiscriminantAnalysis()
	lda.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), lda

def random_forest(train_data, train_labels, test_data, test_labels):
	rfc = RandomForestClassifier(n_estimators=50)
	rfc.fit(train_data, train_labels.ravel())
	return model_evaluate(rfc, test_data, test_labels), rfc

def logistic_regression_mle(train_data, train_labels, test_data, test_labels):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(train_data, train_labels.ravel())
	return model_evaluate(lda, test_data, test_labels), logreg

def write_preds_to_file(file_name, df):
	df.to_csv(file_name, header=_OUTPUT_FILE_HEADER_, index_label="Id")

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data = preprocess_data(train_data, _FIELDS_FILE_, False)

	err, model = random_forest(train_data, train_labels, train_data, train_labels)
	print "###############################################"
	print "Trianing error rate:", err
	print "###############################################"

	# cross-validate
	nrfc = RandomForestClassifier(n_estimators=10)
	kfold = cross_validation.KFold(len(train_data), n_folds=5)
	scores = cross_validation.cross_val_score(nrfc, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
	print "Cross-validation score:", sum(scores)/len(scores)
	
	test_data = read_data(_TEST_FILE_NAME_)
	test_data = preprocess_data(test_data, _FIELDS_FILE_, False)

	preds = pd.DataFrame(model.predict(test_data))
	preds.index = preds.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
	write_preds_to_file(output_fname, preds)


