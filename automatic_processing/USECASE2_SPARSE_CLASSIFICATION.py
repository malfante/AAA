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

import json
import numpy as np
from os.path import isfile
from tools import *
from dataset import Dataset
from config import Config
from recording import Recording
from analyzer import Analyzer
import pickle
import sys
from sklearn import preprocessing
from os import system
from features import FeatureVector
import time
from copy import deepcopy
from sklearn.metrics import confusion_matrix, accuracy_score
from math import ceil

# Range of input arguments
verbatim_range = [0,1,2,3] # O: quiet, 1: settings and main tasks, 2: more details regarding ongoing computation, 3: chats a lot


# Read input arguments
if len(sys.argv) != 3:
    print('This usecase should be use with 2 arguments: ')
    print('\t arg1: configuration file path (i.e. settings_15.json) with the configuration file located in config folder')
    print('\t arg2: verbatim, depending on how chatty you want the system to be. Should be in ', verbatim_range)

    print(sys.argv)
    sys.exit()

setting_file_path = sys.argv[1]
verbatim = sys.argv[2]

# Check input arguments
try:
    verbatim = int(verbatim)
except:
    print('Verbatim argument should be an int between in: ', verbatim_range)
    sys.exit()

if verbatim not in verbatim_range:
    print('Verbatim argument should be an int between in: ', verbatim_range, 'and is ', verbatim)
    sys.exit()

if not isfile(setting_file_path):
    print('There is no file at ', setting_file_path, 'please enter a valid path to a configuration file')
    sys.exit()


# If everything is alright
if verbatim > 0 :
    system('clear')
# Init project with configuration file
config = Config(setting_file_path, verbatim=verbatim)
config.readAndCheck()

# Loading of learning data & labels
learningData = pickle.load(open(config.data_to_analyze['path_to_learning_data'], 'rb'))
learningLabels = pickle.load(open(config.data_to_analyze['path_to_learning_labels'], 'rb'))
testData = pickle.load(open(config.data_to_analyze['path_to_testing_data'], 'rb'))
testLabels = pickle.load(open(config.data_to_analyze['path_to_testing_labels'], 'rb'))
fs = config.data_to_analyze['fs']

#learningData = np.vstack(learningData)
#testData = np.vstack(testData)

## LEARNING STAGE
if verbatim > 0:
    print('\n *** LEARNING ***')
    print('Training data & labels: ',np.shape(learningData), np.shape(learningLabels))

# Transform labels (from str to int from 0 to n_class-1)
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)
if verbatim > 0:
    print('Model will be trained on %d classes'%len(labelEncoder.classes_), np.unique(learningLabelsStd), labelEncoder.classes_)

# Feature extraction for each data
nData = np.shape(learningData)[0]
features = FeatureVector(config, verbatim=verbatim)
t_start_features = time.clock()
learningFeatures = extract_features(config, learningData, features, fs)
t_end_features = time.clock()
if verbatim > 0:
    print('Features have been extracted ', np.shape(learningFeatures), t_end_features - t_start_features)

# Scale features and store scaler
scaler = preprocessing.StandardScaler().fit(learningFeatures)
learningFeatures = scaler.transform(learningFeatures)
if verbatim > 0:
    print('Features have been scaled ', np.shape(learningFeatures))

# Get model from config file
model = deepcopy(config.learning['algo'])
if verbatim > 0:
    print("Learning model is: ", model)

# Xvalidation
CM=list()
acc=list()
sss=config.learning['cv']
if verbatim > 0:
    print("Cross-validation will be performed with: ", sss)
t_start_learning = time.clock()
for (i, (train_index, test_index)) in enumerate(sss.split(learningFeatures, learningLabelsStd)):
    probas = model.fit(learningFeatures[train_index], learningLabelsStd[train_index]).predict_proba(learningFeatures[test_index])
    predictionsStd,_ = getClasses(probas, threshold=None, thresholding=False) # This is cross validation ... no thresholding here
    predictions = [labelEncoder.inverse_transform(s) if s in range(len(labelEncoder.classes_)) else 'notSure' for s in predictionsStd]
    CM.append(confusion_matrix(learningLabels[test_index],predictions, labels=labelEncoder.classes_))
    acc.append(accuracy_score(learningLabels[test_index],predictions))
t_end_learning = time.clock()
if verbatim > 0:
    print("Computation time for learning: ", t_end_learning - t_start_learning)
    print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
    print_cm(np.mean(CM, axis=0),labelEncoder.classes_,hide_zeroes=True,max_str_label_size=2,float_display=False)

# Train model on all data
model = model.fit(learningFeatures, learningLabelsStd)


## TEST 1:
ntestData = np.shape(testData)[0]
if verbatim > 0:
    print('\n\n *** ANALYSIS ***')
    print('Test dataset results', np.shape(testData), np.shape(testLabels))
batch_size = 500
n_batch = ceil(ntestData / batch_size)
testCM = 0
# learningFeatures = np.zeros((nData,),dtype=object)
t_start_testing = time.clock()
for i in range(n_batch):
    # Get the data and the labels in proper shape
    batchTestData = testData[i*batch_size:(i+1)*batch_size]
    batchTestLabels = testLabels[i*batch_size:(i+1)*batch_size]
    if verbatim > 1:
        print("Batch test data: ", np.shape(batchTestData), np.shape(batchTestLabels))
    batchTestLabelsStd = np.array([labelEncoder.transform([s])[0] if s in labelEncoder.classes_ else -1 for s in batchTestLabels])
    # Extract features
    t_start_features_ = time.clock()
    batchTestFeatures = extract_features(config, batchTestData, features, fs)
    t_end_features_ = time.clock()
    if verbatim > 1:
        print('Batch test features have been extracted ', np.shape(batchTestFeatures), t_end_features_ - t_start_features_)
    # Scale features
    batchTestFeatures = scaler.transform(batchTestFeatures)
    if verbatim > 1:
        print('Batch test features have been scaled ', np.shape(batchTestFeatures))
    # Get predictions
    batchTestPredictionsStd,_ = getClasses(model.predict_proba(batchTestFeatures),
                                    threshold=config.features['thresholds'],
                                    thresholding=config.features['thresholding'])
    batchTestPredictions = [labelEncoder.inverse_transform(s) if s in range(len(labelEncoder.classes_)) else 'unknown' for s in batchTestPredictionsStd]
    testCM += confusion_matrix(batchTestLabels,batchTestPredictions, labels=list(labelEncoder.classes_)+['unknown'])
t_end_testing = time.clock()
testAcc = np.sum(np.diag(testCM))/np.sum(testCM)
if verbatim > 0:
    print("Computation time for testing: ", t_end_testing - t_start_testing)
    print('Test results: ', testAcc, ' %')
    print_cm(testCM,list(labelEncoder.classes_)+['unknown'],hide_zeroes=True,max_str_label_size=2,float_display=False)

if verbatim > 0:
    print()
