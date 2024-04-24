#!/bin/bash

# Test for addlog.py
echo -e "\nRunning logic bot..."

python ./logic_bot_runner.py -log "INFO" -games 3 -size 10 -mines 10
