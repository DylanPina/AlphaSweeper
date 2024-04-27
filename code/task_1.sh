#!/bin/bash

python main.py -log "DEBUG" -train_data_file "easy/train" -test_data_file "easy/test" -train_games 10 -test_games 2 -width 9 -height 9 -mines 10
