import time
import models as md
from read_data import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

_TEST_MODE_ = False    # code correctness mode (uses only 100 rows of data to train)

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
	 					"remove_sparse_categorical": False,
	 					"merge_sparse": True,
	 					"sparse_threshold" : 100,
	 					"black_list":  ['18','20','23','25','26','58'],
	 					"nmf": True,
	 					"nmf_out":"nmf_features",
	 					"nmf_read": True
					 }

_HOLDOUT_ = True
_CROSS_VALIDATE_ = False
_K_FOLDS_ = 10


_BLENDING_ = True         # choose one of blending and ensembling
_CREATE_ENSEMBLE_ = False
_MAIN_ESTIMATOR_ = "adaboost"

_CONCAT_INIT_DATA_IN_BLENDER_ = True



def concatenate_blended_frames(train_data, train_data_blended):
	train_data = train_data.reset_index(drop=True)
	train_data_blended = train_data_blended.reset_index(drop=True)
	train_data = pd.concat([train_data, train_data_blended], axis = 1)
	return train_data


if __name__ == "__main__":
	data = read_data(_TRAINING_FILE_NAME_)
	test_data = read_data(_TEST_FILE_NAME_)

	if _TEST_MODE_:
		data = data.iloc[0:10000,:]
		test_data = test_data.iloc[0:10,:]
	#print data.describe()

	train_labels = data.label
	train_labels = train_labels.reshape(train_labels.size, 1)
	train_data = data.drop("label", 1)

	train_data, test_data = \
		preprocess_data(train_data, test_data, data_loader_params)

	#train_data = train_data.drop(['31','32','34','35','50'], axis=1)

	if _BLENDING_ or _HOLDOUT_:
		train_data, holdout, train_labels, holdout_labels = \
					cross_validation.train_test_split(train_data, train_labels, test_size=0.2, random_state=88)

	
	if _CREATE_ENSEMBLE_:
		train_data, train_labels, ensemble = md.create_ensemble(train_data, train_labels)


	# first cross-validate
	model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_]()
	if _CROSS_VALIDATE_ and not _BLENDING_:
		md.cross_validate_model(train_data,train_labels, _K_FOLDS_, model)
	

	if _CREATE_ENSEMBLE_:
		test_data = md.append_test_data(test_data, ensemble)

	if _BLENDING_:
		train_data_blended, holdout_blended, test_data_blended = md.blend_models(_K_FOLDS_, train_data, train_labels, holdout, test_data)
		if _CONCAT_INIT_DATA_IN_BLENDER_:
			print "Blending with original dataset:", train_data.shape, train_data_blended.shape
			train_data = concatenate_blended_frames(train_data, train_data_blended)
			#train_data = train_data.join(train_data_blended)
			print "Blending DONE with original dataset:", train_data.shape

			print "Blending with Holdout set:", holdout.shape, holdout_blended.shape
			holdout = concatenate_blended_frames(holdout, holdout_blended)
			print "Blending DONE with Holdout set:", holdout.shape

			print "Blending with Test set:", test_data.shape, test_data_blended.shape
			test_data = concatenate_blended_frames(test_data, test_data_blended)
			print "Blending with Test set:", test_data.shape
			
		else:
			train_data = train_data_blended
			holdout = holdout_blended
			test_data = test_data_blended

		train_data.to_csv("blended_data.csv")


	## final steps
	# reinstantiate
	model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_]()
	err = md.fit_model(model, train_data, train_labels)
	print "###############################################"
	print "MODEL:", model
	print "Trianing error rate:", err
	print "###############################################"

	if _BLENDING_ or _HOLDOUT_:
		holdout_preds = model.predict(holdout)
		holdout_acc = 1-md.evaluate(holdout_preds, holdout_labels.ravel())
		print "###############################################"
		print "Holdout error rate:", holdout_acc
		print "###############################################"

		## now re-instantiate and train on concatenated holdout + train
		train_data = pd.concat([train_data, holdout], axis=0)
		train_labels = np.concatenate([train_labels, holdout_labels], axis=0)

		model = md._ESTIMATORS_META_[_MAIN_ESTIMATOR_]()
		md.fit_model(model, train_data, train_labels)


	preds = model.predict(test_data)
	preds_df = pd.DataFrame(preds)
	preds_df.index = preds_df.index + 1
	#preds.columns = ['Id', 'Prediction']

	output_fname = _OUTPUT_FILE_NAME_.format(_MAIN_ESTIMATOR_, int(time.time()))

	if not _TEST_MODE_:  # write to disk if not in test mode
		write_preds_to_file(output_fname, preds_df, _OUTPUT_FILE_HEADER_)

		# plot feature importance
		if _MAIN_ESTIMATOR_ == "etc" or _MAIN_ESTIMATOR_ == "rf":
			plt.close()
			names = train_data.columns
			print sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
		             reverse=True)

			feature_importance = model.feature_importances_
			feature_importance = 100.0 * (feature_importance / feature_importance.max())
			sorted_idx = np.argsort(feature_importance)
			pos = np.arange(sorted_idx.shape[0]) + .5
			plt.barh(pos, feature_importance[sorted_idx], align='center')
			plt.yticks(pos, names[sorted_idx])
			plt.title("RF Feature map")
			plt.xlabel("importance")
			plt.ylabel("fname")
			#plt.show()
			plt.savefig("figs/rf_feature_imp.png", format="png")

