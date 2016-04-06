import time
from models import *
from sklearn.ensemble import ExtraTreesClassifier

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"

_CROSS_VALIDATE_ = False

_NORMALIZE_ = True

_K_FOLDS_ = 5

_N_ESTIMATORS_ = [50,150,200,250,300,350,400,450,500]
_MAX_FEATURES_ = ["log2", "None"]
_MAX_DEPTH_ = [4,6,8,10,12,14,16,18,20]

def model_instantiate(estm):
	return ExtraTreesClassifier(n_estimators=estm, n_jobs=-1, random_state=342)

def model_cross_validate(train_data, train_labels, kfolds, estm):
	model = model_instantiate()
	kfold = cross_validation.KFold(len(train_data), n_folds=kfolds)
	scores = cross_validation.cross_val_score(nrfc, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
	print "Cross-validation score:", sum(scores)/len(scores)
	print "###############################################"

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data = preprocess_data(train_data, _FIELDS_FILE_,_NORMALIZE_)

	model = model_instantiate()

	# cross-validate
	if _CROSS_VALIDATE_:
		model_cross_validate(train_data,train_labels, _K_FOLDS_)
	
	test_data = read_data(_TEST_FILE_NAME_)
	test_data = preprocess_data(test_data, _FIELDS_FILE_, _NORMALIZE_)

	preds = model.predict(test_data)
	preds_df = pd.DataFrame(preds)
	preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
	write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)


