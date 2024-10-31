#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate spike count data for MiSO simulation.

This script generates spike count data using average firing rate data files
and a Poisson process.

Author: Yuki Minai
Created: 2024-11-01
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""

import os
import numpy as np
import pandas as pd
import yaml


def generate_spike_count_data(filenames, config_path):
    """
    Generate spike count data with a Poisson process based on mean firing rates.

    Parameters:
        filenames (list of str): List of filenames containing mean firing rate data.
        config_path (str): Path to the configuration file.

    Returns:
        list of pandas.DataFrame: Binned spike count data for each file.
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    input_filepath = config['input'].get('filepath', './')
    binsize_ms = config['spike_data_config'].get('binsize', 50)
    binsize_s = binsize_ms / 1000
    num_no_stim_trials = config['spike_data_config'].get('num_noStim_trials', 0)
    num_stim_trials = config['spike_data_config'].get('num_stim_trials', 0)
    output_filepath = config['output'].get('filepath', './')
    
    # Generate spike count data for each file
    binned_spike_dfs = []
    for file_ind, filename in enumerate(filenames):
        print(f"Processing {filename}")
        
        # Load firing rate data
        fr_df = pd.read_csv(f"{input_filepath}fr_{filename}.csv", header=None)
        
        # Load number of trials to generate for no uStim and uStim trials
        binned_spike_counts = []
        stim_chan = []
        num_no_stim_trial = num_no_stim_trials[file_ind]
        num_stim_trial = num_stim_trials[file_ind]
        
        # Generate spike count data for no uStim trials
        print(f"Generating spike count data for {num_no_stim_trial} non-uStim bins")
        for _ in range(num_no_stim_trial):
            binned_spike_count = [
                np.random.poisson(rate * binsize_s, size=1)[0]
                for rate in fr_df.iloc[0, :]
            ]
            binned_spike_counts.append(binned_spike_count)
            stim_chan.append(0)
        
        # Generate spike count data for uStim trials
        if num_stim_trial != 0:
            print(f"Generating spike count data for {num_stim_trial} uStim bins for each pattern")
            for chan in range(1, 97):  # uStim channels
                for _ in range(num_stim_trial):
                    binned_spike_count = [
                        np.random.poisson(rate * binsize_s, size=1)[0]
                        for rate in fr_df.iloc[chan, :]
                    ]
                    binned_spike_counts.append(binned_spike_count)
                    stim_chan.append(chan)
                
        binned_spike_counts = np.array(binned_spike_counts)
        binned_spike_df = pd.DataFrame(binned_spike_counts)
        stim_chan = np.array(stim_chan)
        binned_spike_df['stim_chan'] = stim_chan
        
        # Rename the column names
        new_columns = np.append(np.arange(1, 97), binned_spike_df.columns[-1])
        binned_spike_df.columns = new_columns
        
        # Save binned spike count data
        output_file = f"{output_filepath}binned_spike_cnt_{filename}.csv"
        print(f"Saving binned spike count data to {output_file}")
        binned_spike_df.to_csv(output_file, index=False)
        binned_spike_dfs.append(binned_spike_df)

    return binned_spike_dfs
