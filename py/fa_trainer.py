#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit Factor Analysis (FA) using binned spike count data.

This script fits a FA model using the EM algorithm for each spike count data file.

Author: Yuki Minai
Created: 2024-11-01
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""

import sys
import numpy as np
import pandas as pd
import pickle
import yaml

sys.path.append("./util/")
from fa import factor_analysis as fa_mdl
from fa_loading_aligner import align_loadings


def fit_fa(filenames, config_path):
    """
    Fit a FA model for each session data.

    Parameters:
        filenames (list of str): List of filenames containing mean firing rate data.
        config_path (str): Path to the configuration file.

    Returns:
        list of fa_model: FA model for each session.
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dimensionality = config['fa'].get('dimensionality', 4)
    model_type = config['fa'].get('model_type', 'fa')
    
    # Fit FA for each session
    for file_ind, filename in enumerate(filenames):
        print(f"Processing {filename}")

        # Load usable channel list
        usable_chan_bool = pd.read_csv(
            f"../data/usable_channels/usable_channel_bool_{filename}.csv", header=None
        )
        usable_chan_bool = np.array(usable_chan_bool).flatten()

        # Load binned spike data
        binned_spike = pd.read_csv(
            f"../output/binned_spike_cnt/binned_spike_cnt_{filename}.csv"
        )
        
        # Extract no uStim trials
        binned_spike_nouStim = binned_spike[binned_spike['stim_chan'] == 0].iloc[:, :-1] # Remove uStim chan column

        # Extract binned spike data of usable channels
        binned_spike_nouStim_usable = binned_spike_nouStim.iloc[:, usable_chan_bool]

        # Fit FA model using no uStim trials
        fa_model = fa_mdl.factor_analysis(model_type=model_type)
        LL, testLL = fa_model.train(
            np.array(binned_spike_nouStim_usable), dimensionality
        )

        # Save FA model        
        fa_save_filename = f"../model/fa/fa_{filename}.pkl"
        print(f"Saving FA model to {fa_save_filename}")
        with open(fa_save_filename, 'wb') as f:
            pickle.dump(fa_model, f)
