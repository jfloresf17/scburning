#!/bin/bash

# This script is used to run the experiment for the paper
python trainer.py ./config.yaml

# nohup ./experiment.sh > unetft.out &