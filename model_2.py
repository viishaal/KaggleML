import time
import models as md
from read_data import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

## DRIVER CONFIG
_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"
_OUTPUT_FILE_NAME_ = "Submissions/{}_{}.csv"
_OUTPUT_FILE_HEADER_ = ["Prediction"]

data_loader_params = {
	 					"ft_file" : _FIELDS_FILE_,
	 					"normalize" : False, 
	 					"one_hot_encode" : True, 
	 					"poly_transform" : True, 
	 					"split_categorical" : False,
	 					"quantize" : True,
	 					"remove_sparse_categorical": True,
	 					"black_list": ['18','20','23','25','26','58']
					 }

_HOLDOUT_ = False
_CROSS_VALIDATE_ = True
_K_FOLDS_ = 10


_BLENDING_ = False         # choose one of blending and ensembling
_CREATE_ENSEMBLE_ = False
_MAIN_ESTIMATOR_ = "etc"


if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	#print data.describe()

	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)
	train_data, transformers = \
		preprocess_data(train_data, data_loader_params, True)

	#train_data = train_data.drop(['31','32','34','35','50'], axis=1)

	if _BLENDING_ or _HOLDOUT_:
		train_data, holdout, train_labels, holdout_labels = \
					cross_validation.train_test_split(train_data, train_labels, test_size=0.2, random_state=88)

	
	if _CREATE_ENSEMBLE_:
		train_data, train_labels, ensemble = md.create_ensemble(train_data, train_labels)

	# err, model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_](train_data, train_labels, train_data, train_labels)

	## plot feature importance
	# plt.close()
	# names = train_data.columns
	# print sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
 #             reverse=True)

	# feature_importance = model.feature_importances_
	# feature_importance = 100.0 * (feature_importance / feature_importance.max())
	# sorted_idx = np.argsort(feature_importance)
	# pos = np.arange(sorted_idx.shape[0]) + .5
	# plt.barh(pos, feature_importance[sorted_idx], align='center')
	# plt.yticks(pos, names[sorted_idx])
	# plt.title("RF Feature map")
	# plt.xlabel("importance")
	# plt.ylabel("fname")
	# #plt.show()
	# plt.savefig("figs/rf_feature_imp.png", format="png")

	# first cross-validate
	model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_]()
	if _CROSS_VALIDATE_:
		md.cross_validate_model(train_data,train_labels, _K_FOLDS_, model)
	
	test_data = read_data(_TEST_FILE_NAME_)
	test_data, _  = \
	 preprocess_data(test_data, data_loader_params, False, transformers)

	#test_data = test_data.drop(['31','32','34','35','50'], axis=1)

	if _CREATE_ENSEMBLE_:
		test_data = md.append_test_data(test_data, ensemble)

	if _BLENDING_:
		train_data, holdout, test_data = md.blend_models(_K_FOLDS_, train_data, train_labels, holdout, test_data)


	## final steps
	# reinstantiate
	model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_]()
	err = md.fit_model(model, train_data, train_labels, None, None)
	print "###############################################"
	print "Trianing error rate:", err
	print "###############################################"

	if _BLENDING_ or _HOLDOUT_:
		holdout_preds = model.predict(holdout)
		holdout_acc = 1-md.evaluate(holdout_preds, holdout_labels.ravel())
		print "###############################################"
		print "Holdout error rate:", holdout_acc
		print "###############################################"


	preds = model.predict(test_data)
	preds_df = pd.DataFrame(preds)
	preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(_MAIN_ESTIMATOR_, int(time.time()))
	write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)

