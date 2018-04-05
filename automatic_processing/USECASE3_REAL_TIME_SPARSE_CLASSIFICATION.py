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
import time
import sys
from features import FeatureVector
from os import system
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing
from DataReadingFunctions import *



# Range of input arguments
verbatim_range = [0,1,2,3] # O: quiet, 1: settings and main tasks, 2: more details regarding ongoing computation, 3: chats a lot
action_range = ['training', 'analyzing']

help = 'This usecase can be used for training or analyzing: \n' + \
        'If training:' +\
        '\n\targ1: configuration file path (i.e. settings_15.json) with the configuration file located in config folder' +\
        '\n\targ2: training' +\
        '\n\targ3: verbatim, depending on how chatty you want the system to be. Should be in [0,1,2,3]' + \
        '\n\nIf analyzing:' +\
        '\n\targ1: configuration file path (i.e. settings_15.json) with the configuration file located in config folder' +\
        '\n\targ2: analyzing' +\
        '\n\targ3: date, with following format yyyy_mm_dd' + \
        '\n\targ4: time, with following format hh_mm_ss.ss' + \
        '\n\targ5: duration, in seconds (can be int or float)' + \
        '\n\targ6: verbatim, depending on how chatty you want the system to be. Should be in [0,1,2,3]'

# Read input arguments
if len(sys.argv) == 4 and sys.argv[2] == 'training':
    setting_file_path = sys.argv[1]
    action = sys.argv[2]
    verbatim = sys.argv[3]
elif len(sys.argv) == 7 and sys.argv[2] == 'analyzing':
    setting_file_path = sys.argv[1]
    action = sys.argv[2]
    date = sys.argv[3]
    time_ = sys.argv[4]
    duration = sys.argv[5]
    verbatim = sys.argv[6]
else:
    print(help)
    print()
    print(sys.argv)
    sys.exit()


# Check input arguments
try:
    verbatim = int(verbatim)
except:
    print('Verbatim argument should be an int between in: ', verbatim_range)
    sys.exit()

if verbatim not in verbatim_range:
    print('Verbatim argument should be an int between in: ', verbatim_range, 'and is ', verbatim)
    sys.exit()

if action not in action_range:
    print('Action argument should be in: ', action_range, 'and is ', action)
    sys.exit()

if not isfile(setting_file_path):
    print('There is no file at ', setting_file_path, 'please enter a valid path to a configuration file')
    sys.exit()

if action == 'analyzing':
    try:
        year,month,day = date.split('_')
        year = int(year)
        month = int(month)
        day = int(day)
    except:
        print('Date should be yyyy_mm_dd')
        print()
        print(help)
        sys.exit()
    try:
        hour,minut,second = time_.split('_')
        hour = int(hour)
        minut = int(minut)
        second = float(second)
    except:
        print('Time should be hh_mm_ss.ss')
        print()
        print(help)
        sys.exit()
    try:
        duration = float(duration)
    except:
        print('Duration should be int or float and is ', duration)
        print()
        print(help)
        sys.exit()
    try:
        tStartSignature = datetime.datetime(year,month,day,hour,minut,int(second),int((second-int(second))*1000000) )
    except Exception as inst:
        print('Problem while reading date or hour : ', year, month, day, hour, minut, second)
        print('--', inst)
        sys.exit()


# If everything is alright
if verbatim > 2 :
    system('clear')
# NB: use of verbatim_system to match wanted use in BPPTKG
verbatim_system = 0
if verbatim > 2:
    verbatim_system=1
# Init project with configuration file
config = Config(setting_file_path, verbatim=verbatim_system)
config.readAndCheck()


# TRAINING THE MODEL
if action == 'training':
    analyzer = Analyzer(config, verbatim=verbatim)
    analyzer.learn(config)
    analyzer.save(config)

# ANALYSIS OF A NEW DATA
if action == 'analyzing':
    # Make or load analyzer (model+scaler)
    analyzer = Analyzer(config, verbatim=verbatim_system)
    analyzer.load(config)

    # Analyzing a new data
    try:
        (fs, signature) = requestObservation(config, tStartSignature, duration, None, verbatim=0)
    except Exception as inst:
        print('Data could not be read')
        print('--', inst)

    # Feature extraction for each data
    my_feature_vector = FeatureVector(config, verbatim=verbatim_system)
    t_start_features = time.time()
    features = extract_features(config, signature.reshape(1, -1), my_feature_vector, fs) #reshape signature, blabla
    t_end_features = time.time()
    if verbatim > 2:
        print('Feature vector has been extracted ', np.shape(features), t_end_features - t_start_features, 'sec')

    # Scale features and store scaler
    t_start_scaler = time.time()
    features = analyzer.scaler.transform(features)
    t_end_scaler = time.time()
    if verbatim > 2:
        print('Feature vector has been scaled ', np.shape(features), t_end_scaler - t_start_scaler, 'sec')

    # Get only the probas:
    t_start_predict = time.time()
    probas = analyzer.model.predict_proba(features)
    t_end_predict = time.time()
    if verbatim > 1:
        print('Output probabilities are: ', t_end_predict - t_start_predict, 'sec')
        for class_name in analyzer.labelEncoder.classes_ :
            print('proba', class_name, '\t', probas[0][analyzer.labelEncoder.transform([class_name])[0]])

    # And take decision if needed
    predictionStd,associated_proba = getClasses(probas,threshold=config.features['thresholds'],thresholding=config.features['thresholding'])
    prediction = [analyzer.labelEncoder.inverse_transform(s) if s in range(len(analyzer.labelEncoder.classes_)) else 'unknown' for s in predictionStd]
    if verbatim > 0:
        print('Prediction result is: ', prediction)
        print('with an associated probability of ', associated_proba)

if verbatim > 2:
    print()
