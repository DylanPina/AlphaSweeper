#!/bin/bash

python test.py -log "DEBUG" -train_data_file "easy/train" -test_data_file "easy/test" -train_games 50000 -test_games 10000 -width 9 -height 9 -mines 10
