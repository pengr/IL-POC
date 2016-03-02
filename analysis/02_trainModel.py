import csv
import math
import sys
import pickle
import random
import time
import os

import numpy as np

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import svm

start_time = time.time()

random.seed(0)

print "loading dataset..."

workingDirectory = os.path.dirname(os.path.abspath(__file__))

with open(workingDirectory + "\\dashilar_dataForModel.pkl", 'rb') as f:
	dataSet = pickle.load(f)

print "dataset loaded [" + str(int(time.time() - start_time)) + " sec]"

random.shuffle(dataSet)

## MACHINE LEARNING IMPLEMENTATION
featureData = []
targetData = []

# use timeframe
# for record in dataSet:
# 		if record[1] <= 31: #limit timeframe to Jan-June
# 			featureData.append(record[:-1])
# 			targetData.append(record[-1])

# use subsample
# for record in dataSet[:25000]:
# 	featureData.append(record[:-1])
# 	targetData.append(record[-1])

# use all data
for record in dataSet:
	featureData.append(record[:-1])
	targetData.append(record[-1])

# data dictionary
# 0 - id
# 1 - dow
# 2 - doy
# 3 - lat
# 4 - lng
# 5 - cat
# 6 - activity


idx_IN_columns = [1, 2, 3, 4, 5] # with weather # with DOW

X = np.asarray(featureData, dtype='float')[:,idx_IN_columns]
y = np.asarray(targetData, dtype='float')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	X, y, test_size=0.3, random_state=0)

print "length of dataset: " + str(X.shape[0])
print "length of training set: " + str(X_train.shape[0])
print "length of test set: " + str(X_test.shape[0])


#mean 0, variance 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

print X_train.shape


# # CROSS VALIDATION ROUTINE

start_time_total = time.time()

rs_max = 0

for C in [1, 100, 10000]:
# for C in [100]:

	for e in [.0001, .01, 1]:
	# for e in [1]:

			for g in [.01, 1, 100]:
			# for g in [1]:

				start_time = time.time()

				model = svm.SVR(C=C, epsilon=e, gamma=g, kernel='rbf', cache_size=8000)
				scores = cross_validation.cross_val_score(model, X_train_scaled, y_train, cv=5)
				rs = scores.mean()

				if rs > rs_max:
					rs_max = rs
					model_best = model
					C_best = C
					e_best = e
					g_best = g

				print "finished model: C[" + str(C) + "], e[" + str(e) + "], g[" + str(g) + "], score[" + str(rs) + "], [" + str(int(time.time() - start_time)) + " sec]"


print "training complete [" + str(int(time.time() - start_time_total)) + " sec]"
print "best model: C[" + str(C_best) + "], e[" + str(e_best) + "], g[" + str(g_best) + "]"



start_time_total = time.time()

model_best.fit(X_train_scaled, y_train)

print "training complete [" + str(int(time.time() - start_time_total)) + " sec]"

X_test_scaled = scaler.transform(X_test)
rs_final = model_best.score(X_test_scaled, y_test)

print "R squared (in test set): " + str(rs_final)


with open(workingDirectory + '\\dataModel.pkl', 'wb') as f:
	pickle.dump(model_best, f)

with open(workingDirectory + '\\scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)