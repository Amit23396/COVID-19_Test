import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[test_indices], data.iloc[test_indices]

if __name__ == '__main__':
	#Read csv file
	df = pd.read_csv("D://data.csv")
	train, test = data_split(df, .2)
	
	#X_train = train['fever', 'bodypain', 'age', 'runnyNose', 'diffBrith']
	X_train = train[list(train)[:-1]].to_numpy()
	X_test = test[list(train)[:-1]].to_numpy()
	
	Y_train = train[['infectionProb']].to_numpy().reshape(725 , )
	Y_test = test[['infectionProb']].to_numpy().reshape(725, )
	
	clf = LogisticRegression()
	clf.fit(X_train, Y_train)

	file = open('model.pkl','wb')
	pickle.dump(clf, file)
	file.close()