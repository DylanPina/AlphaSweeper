#!/bin/bash

python task_1.py -log "DEBUG" -train -train_data_file "easy/train" -test_data_file "easy/test" -train_games 20000 -test_games 5000 -network_bot_games 100 -width 9 -height 9 -mines 10
