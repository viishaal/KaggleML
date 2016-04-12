import numpy as np
linalg = np.linalg
from read_data import *
from sklearn import preprocessing

_TRAINING_FILE_NAME_ = "Data/data.csv"
_TEST_FILE_NAME_ = "Data/quiz.csv"
_FIELDS_FILE_ = "Data/field_types.txt"

BLACKLIST = ['18','20','23','25','26','58']

def independent_columns(A, tol = 1e-05):
    """
    Return an array composed of independent columns of A.

    Note the answer may not be unique; this function returns one of many
    possible answers.

    http://stackoverflow.com/q/13312498/190597 (user1812712)
    http://math.stackexchange.com/a/199132/1140 (Gerry Myerson)
    http://mail.scipy.org/pipermail/numpy-discussion/2008-November/038705.html
        (Anne Archibald)

    >>> A = np.array([(2,4,1,3),(-1,-2,1,0),(0,0,2,2),(3,6,2,5)])
    >>> independent_columns(A)
    np.array([[1, 4],
              [2, 5],
              [3, 6]])
    """
    Q, R = linalg.qr(A)
    independent = np.where(np.abs(R.diagonal()) > tol)[0]
    #print independent
    return independent
    #return A[:, independent]


if __name__ == "__main__":
    data = read_data(_TRAINING_FILE_NAME_)
    #print data.describe()
    train_labels = data.label
    train_labels = train_labels.reshape(train_labels.size, 1)

    train_data = data.drop("label", 1)
    train_data =train_data.drop(BLACKLIST, axis=1)

    ft = open(_FIELDS_FILE_, "r")
    categ = []      # list of categorical variables for transform
    non_categ = []
    categ = []
    for line in ft.readlines():
        splits = line.split()
        if splits[1] == "numeric":
            non_categ.append(splits[0])
        elif splits[0] not in BLACKLIST:
            categ.append(splits[0])

    train_data =train_data.drop(non_categ, axis=1)

    # for i,c in enumerate(categ):
    #     le = preprocessing.LabelEncoder()
    #     train_data[c] = le.fit_transform(train_data[c])


    print train_data.shape

    one_hot_features = []
    for c in categ:
        print c
        lb = preprocessing.LabelBinarizer()
        mat = lb.fit_transform(train_data[c])
        print mat[1:10,:]
        one_hot_features.append(mat)

    train_data = train_data.drop(categ, axis=1)

    for mat in one_hot_features:
        train_data = pd.concat([train_data, pd.DataFrame(mat)], axis=1)

    print train_data.shape

    indx = independent_columns(train_data.as_matrix())
    print len(indx)
    dependent =  [i not in indx for i in range(36)]
    print train_data.columns[dependent]

    train_data.drop(train_data.columns[dependent], axis=1)
    print len(independent_columns(train_data.as_matrix()))


