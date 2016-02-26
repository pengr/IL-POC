from flask import Flask
from flask import render_template
from flask import request
from flask import Response

import json
import time
import sys
import random
import math

import pyorient

from Queue import Queue

from sklearn import preprocessing
from sklearn import svm

import numpy as np
import os

currentDirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"" )

app = Flask(__name__)

q = Queue()

def point_distance(x1, y1, x2, y2):
	return ((x1-x2)**2.0 + (y1-y2)**2.0)**(0.5)
	
def remap(value, min1, max1, min2, max2):
	return float(min2) + (float(value) - float(min1)) * (float(max2) - float(min2)) / (float(max1) - float(min1))

def normalizeArray(inputArray):
	maxVal = 0
	minVal = 100000000000

	for j in range(len(inputArray)):
		for i in range(len(inputArray[j])):
			if inputArray[j][i] > maxVal:
				maxVal = inputArray[j][i]
			if inputArray[j][i] < minVal:
				minVal = inputArray[j][i]

	for j in range(len(inputArray)):
		for i in range(len(inputArray[j])):
			inputArray[j][i] = remap(inputArray[j][i], minVal, maxVal, 0, 1)

	return inputArray
	
def exe01(var1):
	return float(var1)/1
	

# def event_stream():
#	  while True:
#		  result = q.get()
#		  yield 'data: %s\n\n' % str(result)

# @app.route('/eventSource/')
# def sse_source():
#	  return Response(
#			  event_stream(),
#			  mimetype='text/event-stream')

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/2/")
def index2():
	return render_template("index2.html")
	
@app.route("/getData/")
def getData():

	q.put("starting data query...")
	lat1 = str(request.args.get('lat1'))
	lng1 = str(request.args.get('lng1'))
	lat2 = str(request.args.get('lat2'))
	lng2 = str(request.args.get('lng2'))

	w = float(request.args.get('w'))
	h = float(request.args.get('h'))
	cell_size = float(request.args.get('cell_size'))

	analysis = request.args.get('analysis')
	analysisType = request.args.get('analysisType')
	print analysisType
	print analysis
	fileName = "weibo_dashilar_withCounts.txt"

	with open(currentDirectory +  "data\\" + fileName, 'r') as f:
		records = f.readlines()
		records = [x.strip() for x in records]
		titles = records.pop(0).split(';')
	print titles

	print len(records)

	# iterate through data to find minimum and maximum price
	minCount = 1000000000
	maxCount = 0

	for record in records:
		features = record.split(';')
		count = int(features[titles.index('count')])
		if count > maxCount:
			maxCount = count
		if count < minCount:
			minCount = count

	print minCount
	print maxCount


	if analysis == "false":
		q.put('idle')
		return json.dumps(output)

	q.put('starting analysis...')

	numW = int(math.floor(w/cell_size))
	numH = int(math.floor(h/cell_size))

############ machine learning ###################
	## MACHINE LEARNING IMPLEMENTATION
	featureData = []
	targetData = []
	for recDord in records:
		features = record.split(';')		
		featureData.append([float(features[titles.index('lat_wgs')]), float(features[titles.index('lng_wgs')])])
		targetData.append(features[titles.index('count')])

        
	X = np.asarray(featureData, dtype='float')
	y = np.asarray(targetData, dtype='float')

	numListings = len(records)
	breakpoint = int(numListings * .7)

	print "length of dataset: " + str(numListings)
	print "length of training set: " + str(breakpoint)
	print "length of validation set: " + str(numListings-breakpoint)

	# create training and validation set
	X_train = X[:breakpoint]

	X_val = X[breakpoint:]
	######X_val = X[:]


	y_train = y[:breakpoint]
	y_val = y[breakpoint:]
	########y_val = y[:]

	#mean 0, variance 1
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)

	mse_min = 10000000000000000000000

	# add values to arrays in nested loops to test other models
	for C in [2,3]:
		for e in [3]:
				for g in [0.0]:
					q.put("training model: C[" + str(C) + "], e[" + str(e) + "], g[" + str(g) + "]")
					#model = svm.SVR(C=C, epsilon=e, gamma='auto', kernel='rbf', cache_size=2000)
					model = svm.SVR( kernel='rbf', cache_size=2000)
					model.fit(X_train_scaled, y_train)
					y_val_p = [model.predict([i]) for i in X_val]

					mse = 0
					for i in range(len(y_val_p)):
						mse += (y_val_p[i] - y_val[i]) ** 2
					mse /= len(y_val_p)

					if mse < mse_min:
						mse_min = mse
						model_best = model
						C_best = C
						e_best = e
						g_best = g

	q.put("best model: C[" + str(C_best) + "], e[" + str(e_best) + "], g[" + str(g_best) + "]")



############ machine learning ###################
	output = {"type":"FeatureCollection","features":[]}
	for i in range( 0,len(records),1):
		record = records[i]
		features = record.split(';')
		point = {"type":"Feature","properties":{},"geometry":{"type":"Point"}}
		point["id"] = features[titles.index('poiid')]
		point["properties"]["name"] = features[titles.index('title')]
		point["properties"]["address"] = features[titles.index('address')]
		point["properties"]["count"] = features[titles.index('count')]
		point["properties"]["countNorm"] = remap(features[titles.index('count')], minCount, maxCount, 0, 1)
		point["properties"]["popularity"] = random.random ()
		point["geometry"]["coordinates"] = [float(features[titles.index('lat_wgs')]), float(features[titles.index('lng_wgs')])]

		testData = [[float(features[titles.index('lat_wgs')]), float(features[titles.index('lng_wgs')])]]
		X_test = np.asarray(testData, dtype='float')
		X_test_scaled = scaler.transform(X_test)
		point["properties"]["popularity"] = model_best.predict(X_test_scaled)[0]
		
		output["features"].append(point)		

	#exe01(float(features[titles.index('lat_wgs')]), float(features[titles.index('lng_wgs')]))
	print len(X_train)
	return json.dumps(output)

if __name__ == "__main__":
	#app.run(host='0.0.0.0',port=5000,debug=True,threaded=True)
	app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)