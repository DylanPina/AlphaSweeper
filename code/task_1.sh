#!/bin/bash

python main.py -log "CRITICAL" -file "task_1_easy" -games 10 -width 9 -height 9 -mines 10
python main.py -log "CRITICAL" -file "task_1_med" -games 10 -width 16 -height 16 -mines 40
python main.py -log "CRITICAL" -file "task_1_exp" -games 10 -width 30 -height 16 -mines 100
