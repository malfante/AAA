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
from os import system


# Range of input arguments
verbatim_range = [0,1,2,3] # O: quiet, 1: settings and main tasks, 2: more details regarding ongoing computation, 3: chats a lot
action_range = ['training', 'analyzing', 'making_decision', 'display']


# Read input arguments
if len(sys.argv) != 4:
    print('This usecase should be use with 3 arguments: ')
    print('\t arg1: configuration file path (i.e. settings_15.json) with the configuration file located in config folder')
    print('\t arg2: action, to be chosen between ', action_range)
    print('\t Please notice that analyzing should not be called if a model has not been trained previously...')
    print('\t arg3: verbatim, depending on how chatty you want the system to be. Should be in ', verbatim_range)
    sys.exit()

setting_file_path = sys.argv[1]
action = sys.argv[2]
verbatim = sys.argv[3]


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


# If everything is alright
if verbatim > 0 :
    system('clear')

# Init project with configuration file
config = Config(setting_file_path, verbatim=verbatim)
config.readAndCheck()

if action == 'training':
    analyzer = Analyzer(config, verbatim=verbatim)
    analyzer.learn(config)
    analyzer.save(config)

elif action == 'analyzing':
    # Get the analyzer
    analyzer = Analyzer(config, verbatim=verbatim)
    try:
        analyzer.load(config)
    except:
        print('No analyzer has been trained with configuration file ', setting_file_path)
        print('Run script with training action first')
        sys.exit()
    # Analyze the dataset
    analyzedSet = Dataset(config,verbatim=verbatim)
    analyzedSet.analyze(analyzer, config, save=True)

elif action == 'making_decision':
    analyzer = Analyzer(config, verbatim=verbatim)
    try:
        analyzer.load(config)
        analyzedSet = Dataset(config,verbatim=verbatim)
        analyzedSet.makeDecision(config, save=True)
    except:
        print('Training and Analyzing should both have been run with configuration file ', setting_file_path)
        print('Run script with training action and then with analyzing first')
        sys.exit()

elif action == 'display':
    analyzedSet = Dataset(config,verbatim=verbatim)
    analyzer = Analyzer(config, verbatim=verbatim)
    try:
        analyzer.load(config)
    except:
        print('Training, analyzing and making_decision should both have been run with configuration file ', setting_file_path)
        print('Run script with training action and then with analyzing first')
        sys.exit()

    display_for_checking = config.display['display_for_checking']
    if display_for_checking:
        analyzedSet.display(config, onlineDisplay=False, saveDisplay=True, forChecking=True, labelEncoder=analyzer.labelEncoder)
        print('hello')
        if verbatim > 0:
            print('world')
            input("Results have been displayed and saved. Please take time to review them, and press Enter.")

        print('hello world')
        trueLabels,predictedLabels=analyzedSet.getNumericalResults(config, analyzer.labelEncoder)
        print('hello, world')
    else:
        analyzedSet.display(config, onlineDisplay=False, saveDisplay=True, forChecking=False)


if verbatim > 0:
    print()
