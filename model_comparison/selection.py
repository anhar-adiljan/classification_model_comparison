from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# perform multiple folds of cross validation on the given classifier
# return average training and test errors
def experiment(clf, x, y, rate=0.2, n_folds=5):
	train_err = 0.
	test_err = 0.
	for i in range(n_folds):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=rate)
		train_n, test_n = (len(train_x), len(test_x))
		clf = clf.fit(train_x, train_y)
		# calculate error on training set
		pred_y = clf.predict(train_x)
		train_err += float(np.count_nonzero(pred_y - train_y)) / train_n
		# calculate error on test set
		pred_y = clf.predict(test_x)
		test_err += float(np.count_nonzero(pred_y - test_y)) / test_n
	train_err /= n_folds
	test_err /= n_folds
	return [train_err, test_err]

def choose_rf(x, y, rate=0.2):
	_, nf = x.shape
	fs = [1, 2, 4, 6, 8, 12, 16, 20]
	cs = ['gini', 'entropy']
	rfs = [RandomForestClassifier(n_estimators=1024, criterion=c, max_features=f) for f in fs if f <= nf for c in cs]
	errors = np.array([experiment(clf, x, y, rate=rate) for clf in rfs])
	idx = np.argmin(errors[:,1])
	return (rfs[idx], errors[idx,1])

def choose_knn(x, y, rate=0.2):
	ns = len(x)
	step = int(ns * (1-rate)) / 26
	ks = [1]
	for i in range(1,26):
		if step % 2 == 0:
			ks.append(ks[-1] + step)
		else:
			ks.append(ks[-1] + step - 1)	
	ws = ['uniform', 'distance']
	# train KNN classifiers
	knns = [KNeighborsClassifier(n_neighbors=k, weights='uniform') for k in ks for w in ws]
	errors = np.array([experiment(clf, x, y, rate=rate) for clf in knns])
	idx = np.argmin(errors[:,1]) 
	return (knns[idx], errors[idx,1])

def choose_mlp(x, y, rate=0.2):
	hus = [1, 2, 4, 8, 32, 128]
	ms = [0.0, 0.2, 0.5, 0.9]
	anns = [MLPClassifier(hidden_layer_sizes=(hu,), momentum=m) for hu in hus for m in ms]
	errors = np.array([experiment(clf, x, y, rate=rate) for clf in anns])
	idx = np.argmin(errors[:,1])
	return (anns[idx], errors[idx,1])