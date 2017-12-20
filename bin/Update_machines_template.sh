#!/bin/bash

# This script will update your all the machines with the new data

# Replace samples200.txt with your new file
#Random Forest
Preprocessing.py -t samples200.txt -n 40 -p split-eval
Preprocessing.py -t samples200.txt -n 40
Run_randomforest.py -t 1000 -w fit -o samples200_pre.fit.pickled
