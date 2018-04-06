#!/bin/bash

. activate AAA
PWD=$(pwd)
cd /home/wo/AAA/automatic_processing
./USECASE3_REAL_TIME_SPARSE_CLASSIFICATION.py "$@"
. deactivate
cd $PWD
