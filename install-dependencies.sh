#!/bin/bash

#Install kaggle api and download data
pip install kaggle
kaggle datasets download -d juanantoniobarragan/kinematic-calibration-of-surgical-robots
unzip -q kinematic-calibration-of-surgical-robots.zip
#Install deep learning dependencies
pip install -r requirements_torch.txt 
git clone https://github.com/jabarragann/torch-suite.git --recursive
cd torch-suite
git checkout ff68b04
pip install -e .
pip install -e ./pytorch-checkpoint/
cd ..
#Run script to calculate neural net corrected joints
python3 scripts/01_calibration_exp/07_plot_corrected_joints.py -r ./icra2023-data/d04-rec-20-trajsoft/ --testid 3 7 26 -t -m ./icra2023-data/neuralnet/model
