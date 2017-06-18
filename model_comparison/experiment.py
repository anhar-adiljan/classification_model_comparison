from data_loader import *
from selection import *
import numpy as np

def choose_and_print(x,y,r):
	print('Selecting models...')
	rf, erf = choose_rf(x,y,rate=r)
	print('RF: \t training accuracy = {:f}'.format(1-erf))
	
	knn, eknn = choose_knn(x,y,rate=r)
	print('KNN: \t training accuracy = {:f}'.format(1-eknn))
	
	mlp, emlp = choose_mlp(x,y,rate=r)
	print('MLP: \t training accuracy = {:f}'.format(1-emlp))
	
	return (rf, knn, mlp)

def test_accuracy(clf, x, y):
	p = clf.predict(x)
	e = float(np.count_nonzero(p-y)) / len(x)
	return e
	
rounds = 5
rs = [0.2, 0.4, 0.6]
'''
for i in range(rounds):
	print('Round #' + str(i+1))
	for r in rs:
		print('On iris dataset with {:d}/{:d} split...'.format(int(100*r), int(100-100*r)))
		x_tr, x_tst, y_tr, y_tst = load_iris_data()
		rf, knn, mlp = choose_and_print(x_tr, y_tr, r)
		print('Calculating accuracy on test set...')
		erf = test_accuracy(rf, x_tst, y_tst)
		eknn = test_accuracy(knn, x_tst, y_tst)
		emlp = test_accuracy(mlp, x_tst, y_tst)
		print('RF: \t test accuracy = {:f}'.format(1-erf))
		print('KNN: \t test accuracy = {:f}'.format(1-eknn))
		print('MLP: \t test accuracy = {:f}'.format(1-emlp))
		print('')

print('')

for i in range(rounds):
	print('Round #' + str(i+1))
	for r in rs:
		print('On cancer dataset with {:d}/{:d} split...'.format(int(100*r), int(100-100*r)))
		x_tr, x_tst, y_tr, y_tst = load_cancer_data()
		rf, knn, mlp = choose_and_print(x_tr, y_tr, r)
		print('Calculating accuracy on test set...')
		erf = test_accuracy(rf, x_tst, y_tst)
		eknn = test_accuracy(knn, x_tst, y_tst)
		emlp = test_accuracy(mlp, x_tst, y_tst)
		print('RF: \t test accuracy = {:f}'.format(1-erf))
		print('KNN: \t test accuracy = {:f}'.format(1-eknn))
		print('MLP: \t test accuracy = {:f}'.format(1-emlp))
		print('')

print('')
'''
for i in range(rounds):
	print('Round #' + str(i+1))
	for r in rs:
		print('On adult dataset with {:d}/{:d} split...'.format(int(100*r), int(100-100*r)))
		x_tr, x_tst, y_tr, y_tst = load_adult_data()
		rf, knn, mlp = choose_and_print(x_tr, y_tr, r)
		print('Calculating accuracy on test set...')
		erf = test_accuracy(rf, x_tst, y_tst)
		eknn = test_accuracy(knn, x_tst, y_tst)
		emlp = test_accuracy(mlp, x_tst, y_tst)
		print('RF: \t test accuracy = {:f}'.format(1-erf))
		print('KNN: \t test accuracy = {:f}'.format(1-eknn))
		print('MLP: \t test accuracy = {:f}'.format(1-emlp))
		print('')		
		'''
		print('On adult dataset with {:d}/{:d} split...'.format(int(100*r), int(100-100*r)))
		#x, y = load_digits_data()
		choose_and_print(x, y, r)
		'''