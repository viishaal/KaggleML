import pandas as pd
from sklearn import preprocessing
import numpy as np
from utils import *

BLACKLIST = ['18','20','23','25','26','58']
_QUANTIZE_ = True

def read_data(file_name):
	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, field_types_file, isNormalize, oneHot, polyTransform, label_encoders, label_binarizers, poly_transformer, split_categorical):

	# remove blacklisted features
	data = data.drop(BLACKLIST, axis=1)

	# read filed_types file
	ft = open(field_types_file, "r")
	categ = []      # list of categorical variables for transform
	non_categ = []
	for line in ft.readlines():
		splits = line.split()
		if splits[1] != "numeric" and splits[0] not in BLACKLIST:
			categ.append(splits[0])

	print "Processing categorical features", data.shape

	# split categorical
	new_categs = []
	if split_categorical:
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
		if label_encoders == None:
			le = preprocessing.LabelEncoder()
			data[c] = le.fit_transform(data[c])
			new_label_encoders.append(le)
		else:
			data[c] = label_encoders[i].fit_transform(data[c])


	# Normalize numerical features
	for column in data:
		if column not in categ:
			#data[column] = preprocessing.StandardScalar().fit_transform(data[column].reshape(-1,1))
			non_categ.append(column)


	if _QUANTIZE_:
		# do something with 59 and 60
		data["59p60"] = data["59"] + data["60"]
		data["59m60"] = data["59"] - data["60"]
		lab_1 = np.histogram(data["59"], bins=200)[0]
		data["59quant"] = binning(data["59"], 5, lab_1)

		lab_2 = np.histogram(data["60"], bins=200)[0]
		data["60quant"] = binning(data["60"], 5, lab_2)

	if polyTransform:
		print "Adding polynomial features: ", data.shape
		poly_features = ['59', '60']
		data_to_transform = data[poly_features]
		if not poly_transformer:
			poly = preprocessing.PolynomialFeatures(2, include_bias=False)
			transformed_df = poly.fit_transform(data_to_transform)
			#print type(transformed_df), transformed_df.shape
			new_poly = poly
			data = data.drop(poly_features, axis=1) #redundant features
			data = pd.concat([data, pd.DataFrame(transformed_df)], axis=1)
		else:
			transformed_df = poly_transformer.fit_transform(data_to_transform)
			data = data.drop(poly_features, axis=1) #redundant features
			data = pd.concat([data, pd.DataFrame(transformed_df)], axis=1)
		print "Done adding polynomial features: ", data.shape


	if isNormalize:
		data[non_categ] = preprocessing.scale(data[non_categ])

	if oneHot:
		one_hot_features = []
		i = 0
		for c in categ:
			if c!="23" and c!="58":
				#data = pd.concat([data, pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))], axis=1)
				#df = pd.get_dummies(data[c]).rename(columns=lambda x: c + str(x))
				if not label_binarizers:
					lb = preprocessing.LabelBinarizer()
					mat = lb.fit_transform(data[c])
					one_hot_features.append(mat)
					new_lb.append(lb)
					#print i, c, df.shape
				else:
					mat =label_binarizers[i].transform(data[c])
					one_hot_features.append(mat)

				i = i + 1

		for mat in one_hot_features:
			data = pd.concat([data, pd.DataFrame(mat)], axis=1)

	print "Done processing categorical features", data.shape

	#imp = Imputer(missing_values='null', strategy='most_frequent', axis=0)
	#imp.fit(data)
	return data, new_label_encoders, new_lb, new_poly

def write_preds_to_file(file_name, df, _header_):
	df.to_csv(file_name, header=_header_, index_label="Id")