from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import data_loader as dl
import numpy as np

def train_test_rf(clf, x, y, rate):
	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=rate)
	train_n, test_n = (len(train_x), len(test_x))
	print "Training started..."
	clf = clf.fit(train_x, train_y)
	print "Training completed!"
	# calculate error on training set
	print "Testing on training set"
	pred_y = clf.predict(train_x)
	train_err = float(np.count_nonzero(pred_y - train_y)) / train_n
	# calculate error on test set
	print "Testing on test set"
	pred_y = clf.predict(test_x)
	test_err = float(np.count_nonzero(pred_y - test_y)) / test_n
	return (train_err, test_err)
	
dset = dl.load_iris_data()
ns, nf = dset.shape
clf = RandomForestClassifier()
train_err, test_err = train_test_rf(clf, dset[:,1:nf-1], dset[:,nf-1], 0.2)
print train_err
print test_err