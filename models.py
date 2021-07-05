import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from solve import _data
from Input.process.solve import get_dic
from Input.process.process_input_image import _process

# get dictionary to show image name
dic = get_dic()

# get data training
X_train, y_train = _data()

# create models
logis = LogisticRegression(max_iter=1000)
logis.fit(X_train, y_train)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# linear regression
def lr(path):
    X_test = _process(path)
    start = time.time()
    y_pred = logis.predict(X_test)
    end = time.time()
    res = dic[int(y_pred[0])]
    time_predict = round(end-start, 10)
    return res, time_predict

# decision tree
def dt(path):
    X_test = _process(path)
    start = time.time()
    y_pred = regressor.predict(X_test)
    end = time.time()
    res = dic[int(y_pred[0])]
    time_predict = round(end-start, 10)
    return res, time_predict

# Bayes
def nb(path):
    X_test = _process(path)
    start = time.time()
    y_pred = gnb.predict(X_test)
    end = time.time()
    res = dic[int(y_pred[0])]
    time_predict = round(end-start, 10)
    return res, time_predict

# SVM
def svm(path):
    X_test = _process(path)
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    res = dic[int(y_pred[0])]
    time_predict = round(end-start, 10)
    return res, time_predict

