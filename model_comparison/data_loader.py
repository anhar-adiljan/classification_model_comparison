from sklearn.model_selection import train_test_split
import sklearn.datasets as ds
import numpy as np

def adult_encode_categories():
	categories = {
		'workclass': {
			'Private':          np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
			'Self-emp-not-inc': np.array([0, 1, 0, 0, 0, 0, 0, 0]).astype('float'),
			'Self-emp-inc':     np.array([0, 0, 1, 0, 0, 0, 0, 0]).astype('float'),
			'Federal-gov':      np.array([0, 0, 0, 1, 0, 0, 0, 0]).astype('float'),
			'Local-gov':        np.array([0, 0, 0, 0, 1, 0, 0, 0]).astype('float'),
			'State-gov':        np.array([0, 0, 0, 0, 0, 1, 0, 0]).astype('float'),
			'Without-pay':      np.array([0, 0, 0, 0, 0, 0, 1, 0]).astype('float'),
			'Never-worked':     np.array([0, 0, 0, 0, 0, 0, 0, 1]).astype('float')
		},
		
		'education': {
			'Bachelors':    np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'Some-college': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'11th':         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
			'HS-grad':      np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
		    'Prof-school':  np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    'Assoc-acdm':   np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    'Assoc-voc':    np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    '9th':          np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    '7th-8th':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
		    '12th':         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).astype('float'),
		    'Masters':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).astype('float'),
		    '1st-4th':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype('float'), 
		    '10th':         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).astype('float'), 
		    'Doctorate':    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype('float'),
		    '5th-6th':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).astype('float'),
		    'Preschool':    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).astype('float')
		},
		
		'marital-status': {
			'Married-civ-spouse':    np.array([1, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'Divorced':              np.array([0, 1, 0, 0, 0, 0, 0]).astype('float'), 
			'Never-married':         np.array([0, 0, 1, 0, 0, 0, 0]).astype('float'), 
			'Separated':             np.array([0, 0, 0, 1, 0, 0, 0]).astype('float'), 
			'Widowed':               np.array([0, 0, 0, 0, 1, 0, 0]).astype('float'), 
			'Married-spouse-absent': np.array([0, 0, 0, 0, 0, 1, 0]).astype('float'), 
		    'Married-AF-spouse':     np.array([0, 0, 0, 0, 0, 0, 1]).astype('float')
		},                   
		
		'occupation': {
			'Tech-support':      np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'Craft-repair':      np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'Other-service':     np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
			'Sales':             np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
		    'Exec-managerial':   np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    'Prof-specialty':    np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    'Handlers-cleaners': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).astype('float'),
		    'Machine-op-inspct': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).astype('float'), 
		    'Adm-clerical':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).astype('float'), 
		    'Farming-fishing':   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype('float'),
		    'Transport-moving':  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).astype('float'), 
		    'Priv-house-serv':   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype('float'), 
		    'Protective-serv':   np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).astype('float'),
		    'Armed-Forces':      np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).astype('float')
		},
		
		'relationship': {
			'Wife':           np.array([1, 0, 0, 0, 0, 0]).astype('float'), 
			'Own-child':      np.array([0, 1, 0, 0, 0, 0]).astype('float'), 
			'Husband':        np.array([0, 0, 1, 0, 0, 0]).astype('float'),
			'Not-in-family':  np.array([0, 0, 0, 1, 0, 0]).astype('float'),
			'Other-relative': np.array([0, 0, 0, 0, 1, 0]).astype('float'),
			'Unmarried':      np.array([0, 0, 0, 0, 0, 1]).astype('float')
		},
		
		'race': {
			'White':              np.array([1, 0, 0, 0, 0]).astype('float'), 
			'Asian-Pac-Islander': np.array([0, 1, 0, 0, 0]).astype('float'), 
			'Amer-Indian-Eskimo': np.array([0, 0, 1, 0, 0]).astype('float'), 
		    'Other':              np.array([0, 0, 0, 1, 0]).astype('float'),
		    'Black':              np.array([0, 0, 0, 0, 1]).astype('float')
		},
		
		'sex': {
			'Female': np.array([1, 0]).astype('float'), 
			'Male':   np.array([0, 1]).astype('float')
		},
		
		'native-country': {
			'United-States':              np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
			'Cambodia':                   np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
			'England':                    np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
			'Puerto-Rico':                np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'),
		    'Canada':                     np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'),
		    'Germany':                    np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Outlying-US(Guam-USVI-etc)': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'India':                      np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'),
		    'Japan':                      np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Greece':                     np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'South':                      np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'China':                      np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Cuba':                       np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Iran':                       np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Honduras':                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Philippines':                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Italy':                      np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Poland':                     np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Jamaica':                    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Vietnam':                    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Mexico':                     np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Portugal':                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Ireland':                    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'France':                     np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Dominican-Republic':         np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Laos':                       np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Ecuador':                    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Taiwan':                     np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Haiti':                      np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Columbia':                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Hungary':                    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Guatemala':                  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Nicaragua':                  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]).astype('float'), 
		    'Scotland':                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]).astype('float'), 
		    'Thailand':                   np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]).astype('float'), 
		    'Yugoslavia':                 np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]).astype('float'), 
		    'El-Salvador':                np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]).astype('float'), 
		    'Trinadad&Tobago':            np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]).astype('float'), 
		    'Peru':                       np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]).astype('float'), 
		    'Hong':                       np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).astype('float'), 
		    'Holand-Netherlands':         np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).astype('float')
		},
		
		'output': {
			'<=50K': -1,
			'>50K':  +1
		}         
	}
	return categories

def load_adult_data():
	one_hot = adult_encode_categories()
	# preprocessing the training data
	txtset = np.loadtxt('datasets/adult/adult.data.txt', dtype='string', delimiter=', ')
	txtset = remove_unknown(txtset)
	n = len(txtset)
	age = txtset[:,0].astype('float').reshape((n,1))
	workclass = np.array([one_hot['workclass'][s] for s in txtset[:,1]])
	fnlwgt = txtset[:,2].astype('float').reshape((n,1))
	education = np.array([one_hot['education'][s] for s in txtset[:,3]])
	education_num = txtset[:,4].astype('float').reshape((n,1))
	marital_status = np.array([one_hot['marital-status'][s] for s in txtset[:,5]])
	occupation = np.array([one_hot['occupation'][s] for s in txtset[:,6]])
	relationship = np.array([one_hot['relationship'][s] for s in txtset[:,7]])
	race = np.array([one_hot['race'][s] for s in txtset[:,8]])
	sex = np.array([one_hot['sex'][s] for s in txtset[:,9]])
	capital_gain = txtset[:,10].astype('float').reshape((n,1))
	capital_loss = txtset[:,11].astype('float').reshape((n,1))
	hours_per_week = txtset[:,12].astype('float').reshape((n,1))
	native_country = np.array([one_hot['native-country'][s] for s in txtset[:,13]])
	y = np.array([one_hot['output'][s] for s in txtset[:,14]])
	x = np.hstack((age, workclass, fnlwgt, education, education_num, marital_status, \
	                  occupation, relationship, race, sex, capital_gain, capital_loss, \
	                  hours_per_week, native_country))
	x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, train_size=5000)
	return (x_tr, x_tst, y_tr, y_tst)
	
def load_iris_data():
	iris = ds.load_iris()
	x = iris.data
	y = iris.target
	x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=0.3)
	return (x_tr, x_tst, y_tr, y_tst)

def load_cancer_data():	
	cancer = ds.load_breast_cancer()
	x = cancer.data
	y = cancer.target
	x_tr, x_tst, y_tr, y_tst = train_test_split(x, y, test_size=0.3)
	return (x_tr, x_tst, y_tr, y_tst)

def remove_unknown(txtset):
	indices = []
	row, col = txtset.shape
	for r in range(row):
		for c in range(col):
			if txtset[r][c] == '?':
				indices.append(r)
	return np.delete(txtset, indices, 0)