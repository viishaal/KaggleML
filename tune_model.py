import time
from models import *
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from read_data import *

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_NAME_ = "Submissions/tuned_{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

_GENERATE_OUTPUT_FILE_ = False

_NORMALIZE_ = True

_K_FOLDS_ = 5

_ONE_HOT_ENCODING_ = True

_N_ESTIMATORS_ = [150]
_MAX_FEATURES_ = ["log2", "None"]
_MAX_DEPTH_ = range(20, 51, 5)


def model_instantiate(estm, depth):
	return ExtraTreesClassifier(n_estimators=estm, n_jobs=-1, random_state=342, max_depth=depth, max_features=None)

def model_cross_validate(train_data, train_labels, kfolds, model):
	kfold = cross_validation.KFold(len(train_data), n_folds=kfolds)
	scores = cross_validation.cross_val_score(model, train_data, train_labels.ravel(), cv=kfold, n_jobs=-1)
	avg_score = sum(scores)/len(scores)
	print "###############################################"
	print "model:", model
	print "Cross-validation score:", avg_score
	print "###############################################"
	return avg_score

if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()
	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data, _ , _ = preprocess_data(train_data, _FIELDS_FILE_,_NORMALIZE_, False, None, None)

	best_avg_score = 0
	bes_model = None
	for estm in _N_ESTIMATORS_:
		scores = []
		for md in _MAX_DEPTH_:
			model = model_instantiate(estm, md)
			score = model_cross_validate(train_data,train_labels, _K_FOLDS_, model)
			scores.append(score)
			if best_avg_score < score:
				best_avg_score = score
				best_model = model

		plt.close()
		plt.plot(_MAX_DEPTH_, scores)
		#plt.errorbar(_N_ESTIMATORS_, scores, yerr=s_devs)
		axes = plt.gca()
		axes.set_xlim([0, max(_MAX_DEPTH_) + 10])
		plt.title("ETC Tuning - {} Depth".format(str(estm)))
		plt.xlabel("depth")
		plt.ylabel("Accuracy")
		plt.savefig("figs/etc_tuning_{}_estm.png".format(str(estm)), format="png")
	
	if _GENERATE_OUTPUT_FILE_:
		test_data = read_data(_TEST_FILE_NAME_)
		test_data, _ , _ = preprocess_data(test_data, _FIELDS_FILE_, _NORMALIZE_, False, None, None)

		best_model.fit(train_data, train_labels.ravel())
		preds = best_model.predict(test_data)
		preds_df = pd.DataFrame(preds)
		preds_df.index = preds_df.index + 1
		#preds.columns = ['Id', 'Prediction']

		output_fname = _OUTPUT_FILE_NAME_.format(int(time.time()))
		write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)


