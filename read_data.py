import pandas as pd
from sklearn import preprocessing
import numpy as np
from utils import *

def read_data(file_name):
	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, params, train, transformers=None):

	train_transformer = {}

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

	new_label_encoders = []
	new_lb = []
	new_poly = []

	for i,c in enumerate(categ):
		if not transformers:
			le = preprocessing.LabelEncoder()
			data[c] = le.fit_transform(data[c])
			new_label_encoders.append(le)
		else:
			data[c] = transformers["label_encoders"][i].fit_transform(data[c])


	# Normalize numerical features
	for column in data:
		if column not in categ:
			#data[column] = preprocessing.StandardScalar().fit_transform(data[column].reshape(-1,1))
			non_categ.append(column)


	if params["quantize"]:
		# do something with 59 and 60
		data["59p60"] = data["59"] + data["60"]
		data["59m60"] = data["59"] - data["60"]
		lab_1 = np.histogram(data["59"], bins=200)[0]
		data["59quant"] = binning(data["59"], 5, lab_1)

		lab_2 = np.histogram(data["60"], bins=200)[0]
		data["60quant"] = binning(data["60"], 5, lab_2)
		[non_categ.append(l) for l in ["59p60", "59m60", "59quant", "60quant"]]

	if params["poly_transform"]:
		print "Adding polynomial features: ", data.shape
		poly_features = ['59', '60']
		data_to_transform = data[poly_features]
		if not transformers:
			poly = preprocessing.PolynomialFeatures(2, include_bias=False)
			transformed_df = poly.fit_transform(data_to_transform)
			#print type(transformed_df), transformed_df.shape
			new_poly = poly
			data = data.drop(poly_features, axis=1) #redundant features
		else:
			transformed_df = transformers["poly_transformer"].fit_transform(data_to_transform)
			data = data.drop(poly_features, axis=1) #redundant features

		new_names = ["poly_"+str(i) for i in range(transformed_df.shape[1])]
		[non_categ.append(n) for n in new_names]
		data = pd.concat([data, pd.DataFrame(transformed_df, columns = new_names)], axis=1)
		print "Done adding polynomial features: ", data.shape


	if params["normalize"]:
		data[non_categ] = preprocessing.scale(data[non_categ])

	new_categ = []
	if params["one_hot_encode"]:
		one_hot_features = {}
		i = 0
		for c in categ:
			#data = pd.concat([data, pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))], axis=1)
			#df = pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))
			if not transformers:
				lb = preprocessing.LabelBinarizer()
				mat = lb.fit_transform(data[c])
				one_hot_features[c] = mat
				new_lb.append(lb)
				#print i, c, df.shape
			else:
				mat =transformers["label_binarizers"][i].transform(data[c])
				one_hot_features[c] = mat

			i = i + 1

		data = data.drop(categ, axis=1)

		for c, mat in one_hot_features.iteritems():
			new_names = ["oh_"+c+"_"+str(i) for i in range(mat.shape[1])]
			#print pd.DataFrame(mat, columns = new_names).shape
			data = pd.concat([data, pd.DataFrame(mat, columns = new_names)], axis=1)
			[new_categ.append(n) for n in new_names]

		to_drop = []
		col_names = data.columns
		if params["remove_sparse_categorical"]:
			if train:
				for i in range(len(col_names)):
					column = col_names[i]
					if column not in non_categ:
						freq = np.sum(data[column])
						if freq < 60:
							to_drop.append(column)

				print "Number of features being dropped: ", len(to_drop)
				train_transformer["sparse_categorical"] = to_drop
			else:
				to_drop = transformers["sparse_categorical"]

			data = data.drop(to_drop, axis=1)

		categ = [c for c in new_categ if c not in to_drop]

	print "Done processing categorical features", data.shape

	#imp = Imputer(missing_values='null', strategy='most_frequent', axis=0)
	#imp.fit(data)

	if train:
		train_transformer["label_encoders"] = new_label_encoders
		train_transformer["label_binarizers"] = new_lb
		train_transformer["poly_transformer"] = new_poly

	return data, train_transformer

def write_preds_to_file(file_name, df, _header_):
	df.to_csv(file_name, header=_header_, index_label="Id")