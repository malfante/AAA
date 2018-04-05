# -*-coding:Utf-8 -*

# Copyright: Marielle MALFANTE - GIPSA-Lab -
# Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, 38000 Grenoble, France
# (04/2018)
#
# marielle.malfante@gipsa-lab.fr (@gmail.com)
#
# This software is a computer program whose purpose is to automatically
# processing time series (automatic classification, detection). The architecture
# is based on machine learning tools.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL

%reload_ext autoreload
%autoreload 2

import json
import numpy as np
from os.path import isfile
from tools import *
from dataset import Dataset
from config import Config
from recording import Recording
from analyzer import Analyzer
from sklearn import preprocessing
from featuresFunctions import energy, energy_u
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from features import FeatureVector
import time
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from math import sqrt, ceil
from sklearn.metrics import confusion_matrix, accuracy_score


# Change if you want your screen to keep quiet
# 0 = quiet
# 1 = in between
# 2 = detailed information
verbatim = 2

# Init project with configuration file
# config = Config('../config/general/newsettings_20.json', verbatim=verbatim)  #Fish
config = Config('../config/general/newsettings_29.json', verbatim=verbatim)  #Merapi
config.readAndCheck()

# Loading of learning data & labels
fs = config.data_to_analyze['fs']
learningData = pickle.load(open(config.data_to_analyze['path_to_learning_data'], 'rb'))
learningLabels = pickle.load(open(config.data_to_analyze['path_to_learning_labels'], 'rb'))
testData1 = pickle.load(open(config.data_to_analyze['path_to_testing_data'], 'rb'))
testLabels1 = pickle.load(open(config.data_to_analyze['path_to_testing_labels'], 'rb'))

#learningData = np.vstack(learningData)
#testData1 = np.vstack(testData1)

#(ndata,_)=np.shape(learningData)
ndata=np.shape(learningData)[0]
ndata


## LEARNING STAGE
print(np.shape(learningData), np.shape(learningLabels))

# Transform labels (from str to int from 0 to n_class-1)
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)
print(np.unique(learningLabelsStd))
print(labelEncoder.classes_)
print()

# Feature extraction for each data
features = FeatureVector(config, verbatim=verbatim)
t_start_features = time.clock()
learningFeatures = extract_features(config, learningData, features, fs)
t_end_features = time.clock()
print('Features have been extracted ', np.shape(learningFeatures), t_end_features - t_start_features)

# Scale features and store scaler
scaler = preprocessing.StandardScaler().fit(learningFeatures)
learningFeatures = scaler.transform(learningFeatures)
print('Features have been scaled ', np.shape(learningFeatures))

# Get model from config file
model = config.learning['algo']
print("Learning model is: ", model)
# Or manually is you prefer
# model = svm.SVC(C=512, kernel='rbf', gamma=2**-7, class_weight=None, probability=True)
# model = RandomForestClassifier(n_estimators=100,criterion='entropy',bootstrap=False, class_weight=None)

# Xvalidation
CM=list()
acc=list()
sss=config.learning['cv']
print("Cross-validation will be performed with: ", sss)
t_start_learning = time.clock()
for (i, (train_index, test_index)) in enumerate(sss.split(learningFeatures, learningLabelsStd)):
    probas = model.fit(learningFeatures[train_index], learningLabelsStd[train_index]).predict_proba(learningFeatures[test_index])
    predictionsStd,_ = getClasses(probas, threshold=None, thresholding=False) # This is cross validation ... no thresholding here
    predictions = [labelEncoder.inverse_transform(s) if s in range(len(labelEncoder.classes_)) else 'notSure' for s in predictionsStd]
    CM.append(confusion_matrix(learningLabels[test_index],predictions, labels=labelEncoder.classes_))
    acc.append(accuracy_score(learningLabels[test_index],predictions))
t_end_learning = time.clock()
print("Computation time for learning: ", t_end_learning - t_start_learning)
print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
print_cm(np.mean(CM, axis=0),labelEncoder.classes_,hide_zeroes=True,max_str_label_size=2,float_display=False)

# Train model on all data
model = model.fit(learningFeatures, learningLabelsStd)


## TEST 1:
ntestData = np.shape(testData1)[0]
print()
print('-------------------------------------------------------------------------')
print('Test1 dataset results', np.shape(testData1), np.shape(testLabels1))
batch_size = 500
n_batch = ceil(ntestData / batch_size)
testCM = 0
# learningFeatures = np.zeros((nData,),dtype=object)
t_start_testing = time.clock()
for i in range(n_batch):
    # Get the data and the labels in proper shape
    batchTestData = testData1[i*batch_size:(i+1)*batch_size]
    batchTestLabels = testLabels1[i*batch_size:(i+1)*batch_size]
    print("Batch test data: ", np.shape(batchTestData), np.shape(batchTestLabels))
    batchTestLabelsStd = np.array([labelEncoder.transform([s])[0] if s in labelEncoder.classes_ else -1 for s in batchTestLabels])
    # Extract features
    t_start_features_ = time.clock()
    batchTestFeatures = extract_features(config, batchTestData, features, fs)
    t_end_features_ = time.clock()
    print('Batch test features have been extracted ', np.shape(batchTestFeatures), t_end_features_ - t_start_features_)
    # Scale features
    batchTestFeatures = scaler.transform(batchTestFeatures)
    print('Batch test features have been scaled ', np.shape(batchTestFeatures))
    # Get predictions
    batchTestPredictionsStd,_ = getClasses(model.predict_proba(batchTestFeatures),
                                    threshold=config.features['thresholds'],
                                    thresholding=config.features['thresholding'])
    batchTestPredictions = [labelEncoder.inverse_transform(s) if s in range(len(labelEncoder.classes_)) else 'unknown' for s in batchTestPredictionsStd]
    testCM += confusion_matrix(batchTestLabels,batchTestPredictions, labels=list(labelEncoder.classes_)+['unknown'])
t_end_testing = time.clock()
testAcc = np.sum(np.diag(testCM))/np.sum(testCM)
print("Computation time for testing: ", t_end_testing - t_start_testing)
print('Test 1 results: ', testAcc, ' %')
print_cm(testCM,list(labelEncoder.classes_)+['unknown'],hide_zeroes=True,max_str_label_size=2,float_display=False)
