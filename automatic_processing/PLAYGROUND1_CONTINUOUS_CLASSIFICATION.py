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
import pickle

# Change if you want your screen to keep quiet
# 0 = quiet
# 1 = in between
# 2 = detailed information
verbatim = 2

# Init project with configuration file
config = Config('../config/general/newsettings_16.json', verbatim=verbatim)
config.readAndCheck()

# Make or load analyzer (model+scaler)
analyzer = Analyzer(config, verbatim=verbatim)
# allData, allLabels = analyzer.learn(config, returnData=True) # If you want the data
analyzer.learn(config)


# Dataset that needs analyzing (files to analyze are specified in configuration file)
analyzedSet = Dataset(config,verbatim=verbatim)
analyzedSet.analyze(analyzer, config, save=True)
analyzedSet.makeDecision(config, save=True)
# --> FOR CONTINUOUS ANALYSIS
# analyzedSet.display(config, onlineDisplay=False, saveDisplay=True, forChecking=False) # Add for checking here
# --> OR FOR CONTINUOUS ANALYSIS THAT WILL NEED CHECKING
analyzedSet.display(config, onlineDisplay=False, saveDisplay=True, forChecking=True, labelEncoder=analyzer.labelEncoder)
# Get the real annotations after reviewing
trueLabels,predictedLabels=analyzedSet.getNumericalResults(config, analyzer.labelEncoder)
