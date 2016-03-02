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

# column reference in dataset
# 0 - id
# 1 - dow
# 2 - doy
# 3 - lat
# 4 - lng
# 5 - cat
# 6 - activity

## MACHINE LEARNING IMPLEMENTATION

# split data into feature and target sets. 
# in this case the target of the model is the activity, which is the last column
featureData = []
targetData = []
for record in dataSet:
	featureData.append(record[:-1])
	targetData.append(record[-1])

# select which columns to include in feature set
# in this case we are ignoring the 'id' column
idx_IN_columns = [1, 2, 3, 4, 5] # with weather # with DOW

# convert data sets to numpy arrays, and filter by columns
X = np.asarray(featureData, dtype='float')[:,idx_IN_columns]
y = np.asarray(targetData, dtype='float')

# split data into training and testing sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
	X, y, test_size=0.3, random_state=0)


print "length of dataset: " + str(X.shape[0])
print "length of training set: " + str(X_train.shape[0])
print "length of test set: " + str(X_test.shape[0])


# scale feature data to mean 0, variance 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)



# CROSS VALIDATION ROUTINE

start_time_total = time.time()

# initialize variable to store best performance number
rs_max = 0

# triple loop to iterate over range of options of 3 hyperparameters

# iterate over range of 'C' variable
for C in [1, 100, 10000]:

	# iterate over range of 'e' variable
	for e in [.0001, .01, 1]:

			# iterate over range of 'g' variable
			for g in [.01, 1, 100]:

				start_time = time.time()

				# set up model with current hyperparameters
				model = svm.SVR(C=C, epsilon=e, gamma=g, kernel='rbf', cache_size=8000)

				# generate model performance using cross_validation training
				scores = cross_validation.cross_val_score(model, X_train_scaled, y_train, cv=5)

				# generate average performance score
				rs = scores.mean()

				# if model score better than current best, store best model and its hyperparameters
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

# fit best model using whole training set
model_best.fit(X_train_scaled, y_train)

print "training complete [" + str(int(time.time() - start_time_total)) + " sec]"

# transform test feature set using stored scaler model
X_test_scaled = scaler.transform(X_test)

# generate final performance score
rs_final = model_best.score(X_test_scaled, y_test)

print "R squared (in test set): " + str(rs_final)

# export best model and scaler for future use
with open(workingDirectory + '\\dataModel.pkl', 'wb') as f:
	pickle.dump(model_best, f)

with open(workingDirectory + '\\scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)