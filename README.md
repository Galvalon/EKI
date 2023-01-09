# EKI (Ensemble Kalman Inversion)

This repository implements the Discrete Ensemble Kalman Inversion used in https://arxiv.org/abs/2107.14508.
It also implements some analysis functions for my master's thesis.

## Install
Clone the repository and install all dependencies via

    pip3 install -r requirements.txt


## Run
First, configure all needed parameters in *config.py*.

To run all analysis functions, simply run the *analysis.py* file after setting the configuration via

    python3 analysis.py
    
## Results
Resulting plots and .csv files will be saved into *plots/*.
This folder contains all plots and tables used in my thesis.
