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

# For Atom, comment otherwise
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
from DataReadingFunctions import *


# Change if you want your screen to keep quiet
# 0 = quiet
# 1 = in between
# 2 = detailed information
verbatim = 3

# Init project with configuration file
config = Config('../config/general/newsettings_31.json', verbatim=verbatim)
config.readAndCheck()

# Make or load analyzer (model+scaler)
analyzer = Analyzer(config, verbatim=verbatim)
analyzer.learn(config)
analyzer.save(config)

# Make or load analyzer (model+scaler)
analyzer = Analyzer(config, verbatim=verbatim)
analyzer.load(config)

# Analyzing a new data
tStartSignature = datetime.datetime(2017,6,6,18,29,int(12.25))
duration = 14.88
label = 'ROCKFALL'
(fs, signature) = requestObservation(config, tStartSignature, duration, None, verbatim=0)

# Feature extraction for each data
my_feature_vector = FeatureVector(config, verbatim=verbatim)
t_start_features = time.time()
features = extract_features(config, signature.reshape(1, -1), my_feature_vector, fs) #reshape signature, blabla
t_end_features = time.time()
print('Feature vector has been extracted ', np.shape(features), t_end_features - t_start_features,'sec')

# Scale features and store scaler
# scaler = preprocessing.StandardScaler().fit(features)
features = analyzer.scaler.transform(features)
print('Feature vector has been scaled ', np.shape(features))

# Get only the probas:
probas = analyzer.model.predict_proba(features)
print('Output probabilities are: ')
for class_name in analyzer.labelEncoder.classes_ :
    print('proba', class_name, '\t', probas[0][analyzer.labelEncoder.transform([class_name])[0]])

# And take decision if needed
predictionStd,associated_proba = getClasses(probas,threshold=config.features['thresholds'],thresholding=config.features['thresholding'])
prediction = [analyzer.labelEncoder.inverse_transform(s) if s in range(len(analyzer.labelEncoder.classes_)) else 'unknown' for s in predictionStd]
print('Prediction result is: ', prediction)
print('with an associated probability of ', associated_proba)

# labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
# [labelEncoder.inverse_transform(s) if s in range(len(labelEncoder.classes_)) else 'notSure' for s in predictedLabels]
# np.array([labelEncoder.transform([s])[0] if s in labelEncoder.classes_ else -1 for s in batchTestLabels])
