import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
import time

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_OUTPUT_FILE_NAME_ = "Submissions/{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

def read_data(file_name):
	data = pd.read_csv(file_name)
	sex = preprocessing.LabelEncoder()
	city = preprocessing.LabelEncoder()
	build = preprocessing.LabelEncoder()
	race = preprocessing.LabelEncoder()
	pct = preprocessing.LabelEncoder()

	data.sex = sex.fit_transform(data.sex)
	data.city = city.fit_transform(data.city)
	data.build = build.fit_transform(data.build)
	data.race = race.fit_transform(data.race)
	data.pct = pct.fit_transform(data.pct)

	return data

def evaluate(predictions, labels):	
	total_examples = labels.size
	return (total_examples - np.sum(predictions == labels))*1.0/total_examples

def model_evaluate(model, test_data, test_labels):
	preds = model.predict(test_data)
	return evaluate(preds.reshape(preds.size, 1), test_labels)

def logistic_regression_mle(train_data, train_labels, test_data, test_labels):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(train_data, train_labels.ravel())
	return model_evaluate(logreg, test_data, test_labels), logreg

def write_preds_to_file(file_name, df):
	df.to_csv(file_name, header=_OUTPUT_FILE_HEADER_, index_label="Id")

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)
	err, model = logistic_regression_mle(train_data, train_labels, train_data, train_labels)

	print "Trianing error rate:", err
	
	test_data = read_data(_TEST_FILE_NAME_)
	preds = pd.DataFrame(model.predict(test_data))
	preds.index = preds.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
	write_preds_to_file(output_fname, preds)


