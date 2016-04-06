import pandas as pd
from sklearn import preprocessing

def read_data(file_name):
	data = pd.read_csv(file_name)
	return data

def preprocess_data(data, field_types_file, isNormalize, oneHot, label_encoders, label_binarizers):

	# read filed_types file
	ft = open(field_types_file, "r")
	categ = []      # list of categorical variables for transform
	non_categ = []
	for line in ft.readlines():
		splits = line.split()
		if splits[1] != "numeric":
			categ.append(splits[0])

	print "Processing categorical feaatures", data.shape

	new_label_encoders = []
	new_lb = []

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
	return data, new_label_encoders, new_lb

def write_preds_to_file(file_name, df, _header_):
	df.to_csv(file_name, header=_header_, index_label="Id")