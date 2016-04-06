import time
from models import *

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_NAME_ = "Submissions/{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

_CROSS_VALIDATE_ = True

_NORMALIZE_ = True
_CREATE_ENSEMBLE_ = True

_K_FOLDS_ = 5

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data = preprocess_data(train_data, _FIELDS_FILE_,_NORMALIZE_)

	if _CREATE_ENSEMBLE_:
		train_data, train_labels, ensemble = create_ensemble(train_data, train_labels)

	err, model = ada_boost_classifier(train_data, train_labels, train_data, train_labels)
	print "###############################################"
	print "Trianing error rate:", err
	print "###############################################"

	# cross-validate
	if _CROSS_VALIDATE_:
		crossValidate_adaboost(train_data,train_labels, _K_FOLDS_)
	
	test_data = read_data(_TEST_FILE_NAME_)
	test_data = preprocess_data(test_data, _FIELDS_FILE_, _NORMALIZE_)

	if _CREATE_ENSEMBLE_:
		test_data = append_test_data(test_data, ensemble)

	preds = model.predict(test_data)
	preds_df = pd.DataFrame(preds)
	preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
	write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)


