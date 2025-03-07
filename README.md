# Multiclass Flare Prediction
Related fields: solar flare prediction, multi-classification, space weather forecasting

Command Example,   
python -m Multi-Flare-prediction.scripts.optimization  

Please see the config file in .scripts/configs/*.yaml

### [Abstract]
In this project, we implement a multi-classification (4 classes) task for solar flare forecasting using line-of-sight magnetograms. The goal of this project is comparing conformal prediction to bayesian uncertainty quantification of Monte-calro drop out or deep ensemble.

### [Experimental design]
1. Data
    1. Type: Magnetogram
    2. Size: 512x512
    3. Dataset span: 2010-2018
    4. Train: Jan-Sep / Test: Oct-Dec

2. Optimization
    1. optimizer: SGD
    2. loss: cross-entropy
    3. scheduler: OneCycleLR (pytorch)

2. Uncertainty quantification
    1. Conformal prediction
        1. Least ambiguous with bounded error levels
        2. Adaptive Prediction Sets (APS)  
        ref: https://mapie.readthedocs.io/en/latest/theoretical_description_classification.html#
    2. Bayesian approaches 
        1. Monte-carlo drop-out
        2. deep ensemble 

        
