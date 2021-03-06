import pandas as pd
from sklearn import preprocessing
import numpy as np
from utils import *
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.decomposition import PCA

def read_data(file_name):
	"""
	Function to read the data file into the memory

	Args:
		file_name: name/path to the file to be read into the memory 

	Returns:
		data: pandas dataframe containing all the data

	"""

	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, test_data, params):
	"""
	Preprocess the training and the test data based on the given parameters

	Args:
		data: training data as pandas dataframe
		test_data: test_data as pandas dataframe
		params: map which specifies the parameters and options/switches to process the dataset before blending/training
			params = {
	 					"ft_file" : file containing information about the fields read
	 					"normalize" : 
	 					"one_hot_encode" : one hot encoding of categorical features
	 					"poly_transform" : add polynomial features
	 					"split_categorical" : split categorical features
	 					"quantize" : bins numeric features
	 					"remove_sparse_categorical": remove sparse categorical features based on sparse threshold
	 					"merge_sparse": merges sparse features instead of removing them
	 					"sparse_threshold" : 100,
	 					"black_list":  list of features to remove from the dataset
	 					"nmf": perform Non-negative Matrix Factorization
	 					"nmf_out": NMF output file name
	 					"nmf_read": Read NMF features from a file on disk
	 					"return_pca": perform PCA on the training data
			}

	Returns:
		train_data: training data
		test_data:  test data
		pca_train_data: pca transformation of training data
		pca_test_data: pca transformation of test data

	"""

	print "train data shape ", data.shape
	print "test_data shape", test_data.shape

	#print data.iloc[0:9,:]
	breakpoint = data.shape[0]
	print "breakpoint: ", breakpoint

	#print test_data.iloc[test_data.shape[0] - 1, :]
	data = pd.concat([data, test_data])
	print "after concat", data.shape

	end = data.shape[0]
	#print data.iloc[0:9,:]
	print "end index:", end

	data = data.reset_index(drop=True)

	# remove blacklisted features
	if params["black_list"] is not None:
		data = data.drop(params["black_list"], axis=1)

	# read filed_types file
	ft = open(params["ft_file"], "r")
	categ = []      # list of categorical variables for transform
	non_categ = []
	for line in ft.readlines():
		splits = line.split()
		if splits[1] != "numeric":
			if params["black_list"] is not None: 
				if splits[0] not in params["black_list"]:
					categ.append(splits[0])
			else: 
				categ.append(splits[0])

	print "Processing categorical features", data.shape

	# split categorical
	new_categs = []
	if params["split_categorical"]:
		for c in categ:
			splits = data[c].str.split("_")
			if not pd.isnull(splits.str[1]).all():   ## dont add fields which dont split
				name_1 = c+"_split__1"
				data[name_1] = splits.str[0]
				new_categs.append(name_1)
				#print c, data[name_1][1:10]

			if not pd.isnull(splits.str[1]).any():   ## for now add only fully splittable fields
				name_2 = c+"_split__2"
				data[name_2] = splits.str[1]
				new_categs.append(name_2)
				#print c, data[name_2][1:10]

		## add to original list
		[categ.append(nf) for nf in new_categs]
		print "New categorical variables added", data.shape


	for i,c in enumerate(categ):
		le = preprocessing.LabelEncoder()
		data[c] = le.fit_transform(data[c])


	# Normalize numerical features
	for column in data:
		if column not in categ:
			#data[column] = preprocessing.StandardScalar().fit_transform(data[column].reshape(-1,1))
			non_categ.append(column)


	if params["quantize"]:
		# do something with 59 and 60
		data["59p60"] = data["59"] + data["60"]
		data["59m60"] = data["59"] - data["60"]
		data["59d60"] = data["59"] / (data["60"] + 1 - np.min(data["60"]))

		sq59_p_sq60 = ( data["59"] ** 2 ) +  ( data["60"] ** 2 )
		#data["59sq_p_60sq"] = ( data["59"] ** 2 ) +  ( data["60"] ** 2 )
		#data["59sq_m_60sq"] = ( data["59"] ** 2 ) -  ( data["60"] ** 2 )
		[non_categ.append(l) for l in ["59p60", "59m60", "59d60"]]
		#[non_categ.append(l) for l in ["59sq_p_60sq", "59sq_m_60sq"]]

		# log features

		data["59log"] = np.log(data["59"] + 1 - np.min(data["59"])).astype("float64")
		data["60log"] = np.log(np.absolute(data["60"]) + 1 - np.min(data["60"])).astype("float64")
		data["59sq_p_60sqlog"] = np.log(sq59_p_sq60 + 1 - np.min(sq59_p_sq60)).astype("float64")
		#non_categ.append("59sq_p_60sqlog")
		[non_categ.append(l) for l in ["59log", "60log", "59sq_p_60sqlog"]]


		data["59sqrt"] = np.sqrt(data["59"] + 1 - np.min(data["59"]))
		data["60sqrt"] = np.sqrt(data["60"] + 1 - np.min(data["60"]))
		[non_categ.append(l) for l in ["59sqrt", "60sqrt"]]
		


		bins = [150,200,250]
		for binw in bins:
			lab_1 = np.histogram(data["59"], bins=binw)[0]
			name1 = "59quant_{}".format(binw)
			data[name1] = binning(data["59"], 5, lab_1)

			lab_2 = np.histogram(data["60"], bins=binw)[0]
			name2 = "60quant_{}".format(binw)
			data["60quant_{}".format(binw)] = binning(data["60"], 5, lab_2)

			non_categ.append(name1)
			non_categ.append(name2)


	if params["poly_transform"]:
		print "Adding polynomial features: ", data.shape
		poly_features = ['59', '60']
		data_to_transform = data[poly_features]

		poly = preprocessing.PolynomialFeatures(3, include_bias=False)
		transformed_df = poly.fit_transform(data_to_transform)
		#print type(transformed_df), transformed_df.shape
		data = data.drop(poly_features, axis=1) #redundant features
		[non_categ.remove(l) for l in poly_features]

		new_names = ["poly_"+str(i) for i in range(transformed_df.shape[1])]
		[non_categ.append(n) for n in new_names]
		data = pd.concat([data, pd.DataFrame(transformed_df, columns = new_names)], axis=1)
		print "Done adding polynomial features: ", data.shape


	if params["normalize"]:
		data[non_categ] = preprocessing.scale(data[non_categ])

	new_categ = []
	if params["one_hot_encode"]:
		print "One hot encoding:", data.shape
		one_hot_features = {}
		for c in categ:
			#data = pd.concat([data, pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))], axis=1)
			#df = pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))

			lb = preprocessing.LabelBinarizer()
			mat = lb.fit_transform(data[c])
			one_hot_features[c] = mat
			#print i, c, df.shape


		data = data.drop(categ, axis=1)

		# concatenate new matrices
		for c, mat in one_hot_features.iteritems():

			# merge sparse features
			if params["merge_sparse"]:
				col_sum = mat.sum(axis=0)
				to_merge_cols = np.where(col_sum < params["sparse_threshold"])[0]
				if len(to_merge_cols) > 0:
					to_merge = mat[:, to_merge_cols]
					to_merge = to_merge.sum(axis=1)
					to_merge = to_merge.reshape(to_merge.shape[0], 1)
					mat = np.delete(mat, to_merge_cols, axis=1)
					mat = np.hstack((mat, to_merge))

			# add new coloumn names and merge with origin data
			new_names = ["oh_"+c+"_"+str(i) for i in range(mat.shape[1])]
			#print pd.DataFrame(mat, columns = new_names).shape
			data = pd.concat([data, pd.DataFrame(mat, columns = new_names)], axis=1)
			[new_categ.append(n) for n in new_names]

		# drop sparse features (use in case do not wish to merge)
		to_drop = []
		col_names = data.columns
		if params["remove_sparse_categorical"]:
			for i in range(len(col_names)):
				column = col_names[i]
				if column not in non_categ:
					freq = np.sum(data[column])
					if freq < params["sparse_threshold"]:
						to_drop.append(column)


			print "Number of features being dropped: ", len(to_drop)
			data = data.drop(to_drop, axis=1)

		categ = [c for c in new_categ if c not in to_drop]
		print "One hot encoding DONE:", data.shape

		if params["nmf"]:
			print "Adding NMF features: ", data.shape
			if params["nmf_read"]:
				transformed_df = pd.read_csv(params["nmf_out"])
				data = pd.concat([data, transformed_df], axis=1)
			else:
				nmf_features = data[categ]
				model1 = NMF(n_components=nmf_features.shape[1], init='nndsvd', random_state=0)
				bin2 = model1.fit_transform(nmf_features.transpose())
				model1.reconstruction_err_ 
				transformed = (model1.components_).transpose()
				transformed_df = pd.DataFrame(transformed)
				transformed_df.to_csv(params["nmf_out"])
				data = data.join(transformed_df)

	pca_train_data, pca_test_data = None, None
	if params["return_pca"]:
		pca = PCA()
		pca_features = data[categ]
		pca.fit(pca_features)
		components = pca.components_

		## print information
		print "PCA components: ", components.shape
		print pca.explained_variance_

		transformed_pca = np.dot(pca_features, components.T)
		transformed_pca = data[non_categ].join(pd.DataFrame(transformed_pca))
		pca_train_data = transformed_pca.iloc[0:breakpoint, :]
		pca_test_data = transformed_pca.iloc[breakpoint:end, :]

	print "Done processing categorical features", data.shape

	#imp = Imputer(missing_values='null', strategy='most_frequent', axis=0)
	#imp.fit(data)

	train_data = data.iloc[0:breakpoint, :]
	print "after breaking train: ", train_data.shape

	test_data = data.iloc[breakpoint:end, :]
	#print test_data.iloc[test_data.shape[0]-1,:]
	return train_data, test_data, pca_train_data, pca_test_data

def write_preds_to_file(file_name, df, _header_):
	"""
		Writes a kaggle style submission file with headers in csv format

		Id,Predictions
		1,0
		2,1
		...

	Args:
		file_name: output file name
		df: output dataframe to write as csv

	Returns:
		None

	"""

	df.to_csv(file_name, header=_header_, index_label="Id")

