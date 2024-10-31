#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a closed-loop experiment session.

This script runs a closed-loop experiment session with MiSO
using the CNN predictions of uStim responses.

Author: Yuki Minai
Created: 2024-11-01
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""

import numpy as np
import pandas as pd
import yaml
import pickle
import copy
import random
import matplotlib.pyplot as plt
import sys

sys.path.append("./util/")
from fa import factor_analysis as fa_mdl
from spike_data_generator import generate_spike_count_data
from fa_trainer import fit_fa
from fa_loading_aligner import align_loadings


def run_closed_loop_experiment(config_path_closed_loop, config_path_spike_cnt, config_path_fa, filename_ref, filename_test):
    """
    Run a closed-loop experiment with MiSO.

    Parameters:
        config_path_closed_loop (str): Path to the closed-loop configuration file.
        config_path_spike_cnt (str): Path to the spike cnt configuration file.
        config_path_fa (str): Path to the fa configuration file.
        filename_ref (list of str): List of a reference session file name.
        filename_test (list of str): List of a test session file name.

    Returns:
        dict: Closed-loop experiment performance summary.
    """
    
    print('Load closed-loop experiment config')
    with open(config_path_closed_loop, 'r') as f:
        config_closed_loop = yaml.safe_load(f)
        
    input_path_base = config_closed_loop['input'].get('input_path_base', './')
    prediction_filename = config_closed_loop['prediction'].get('prediction_filename', '')
    target = config_closed_loop['closed_loop'].get('target', [0, 0])
    target_dimension = config_closed_loop['closed_loop'].get('target_dimension', [0,1])
    bin_size_ms = config_closed_loop['closed_loop'].get('bin_size_ms', 50)
    lr = config_closed_loop['closed_loop'].get('lr', 0.1)
    num_trials = config_closed_loop['closed_loop'].get('num_trials', 1000)
    epsilon = config_closed_loop['closed_loop'].get('epsilon', 0.05)

    # Load test firing rate data
    fr_test_df = np.loadtxt(f"../data/fr/fr_{filename_test[0]}.csv", delimiter=',')

    print('Generate spike count data for calibration')
    generate_spike_count_data(filename_test, config_path_spike_cnt)
        
    print('Fit FA using calibration data')
    fit_fa(filename_test, config_path_fa)
    
    print('Align FA loading to a reference session')
    with open(f"../model/fa/fa_{filename_ref[0]}.pkl", 'rb') as f:
        fa_model_reference = pickle.load(f)
    with open(f"../model/fa/fa_{filename_test[0]}.pkl", 'rb') as f:
        fa_model_test = pickle.load(f)
    
    # Load usable channels
    usable_chan_bool_test = pd.read_csv(f"../data/usable_channels/usable_channel_bool_{filename_test[0]}.csv", header=None)[0]
    usable_chan_bool_reference = pd.read_csv(f"../data/usable_channels/usable_channel_bool_{filename_ref[0]}.csv", header=None)[0]
    usable_chan_list_test = pd.read_csv(f"../data/usable_channels/usable_channel_list_{filename_test[0]}.csv", header=None)[0]
    usable_chan_list_reference = pd.read_csv(f"../data/usable_channels/usable_channel_list_{filename_ref[0]}.csv", header=None)[0]
    usable_chan1 = usable_chan_list_reference
    usable_chan2 = usable_chan_list_test

    # Align FA models
    L1 = fa_model_reference.fa_params['L'] # reference session
    L2 = fa_model_test.fa_params['L'] # align target session
    aligned_L2 = align_loadings(L1, L2, usable_chan1, usable_chan2)    
    fa_model_aligned_test = copy.deepcopy(fa_model_test)
    fa_model_aligned_test.fa_params['L'] = aligned_L2
    
    print('Load CNN prediction')
    predictions = np.loadtxt(f"../output/cnn_prediction/prediction_{prediction_filename}.csv", delimiter=',')

    print(f"Run closed-loop optimization for {num_trials} trials for each method")
    selected_chan_ind, induced_activities_MiSO, induced_activities_noStim, induced_activities_Random = [], [], [], []
    bin_size_s = bin_size_ms/1000

    # Choose the first pattern
    l1_loss = np.abs(predictions - target).sum(axis=1)
    min_loss_stim_chan = np.argmin(l1_loss)

    # Run closed-loop optimization
    for i in range(num_trials):
        # Observe binned spike count data with a uStim pattern selected by MiSO
        binned_spike_count_MiSO = []
        for j, rate in enumerate(fr_test_df[min_loss_stim_chan+1,:]):
            expected_spikes_per_bin = rate * bin_size_s
            binned_spike_count_MiSO.append(np.random.poisson(expected_spikes_per_bin, size=1)[0])
            
        # Observe binned spike count data with non-uStim
        binned_spike_count_noStim = []
        for j, rate in enumerate(fr_test_df[0,:]):
            expected_spikes_per_bin = rate * bin_size_s
            binned_spike_count_noStim.append(np.random.poisson(expected_spikes_per_bin, size=1)[0])
            
        # Observe binned spike count data with Random uStim
        binned_spike_count_Random = []
        random_stim_chan = np.random.choice(np.arange(1,97,1), size=1)[0]
        for j, rate in enumerate(fr_test_df[random_stim_chan,:]):
            expected_spikes_per_bin = rate * bin_size_s
            binned_spike_count_Random.append(np.random.poisson(expected_spikes_per_bin, size=1)[0])
        
        # Compute the latent activity for each observation
        z_MiSO, LL_fit = fa_model_aligned_test.estep(np.array(binned_spike_count_MiSO).reshape(1,96)[:,usable_chan_bool_test]) # USE ALL No uStim TRIALS
        z_MiSO = z_MiSO['z_mu']
        z_MiSO_targetdim = z_MiSO[0,target_dimension]
        
        z_noStim, LL_fit = fa_model_aligned_test.estep(np.array(binned_spike_count_noStim).reshape(1,96)[:,usable_chan_bool_test]) # USE ALL No uStim TRIALS
        z_noStim = z_noStim['z_mu']
        z_noStim_targetdim = z_noStim[0,target_dimension]
        
        z_Random, LL_fit = fa_model_aligned_test.estep(np.array(binned_spike_count_Random).reshape(1,96)[:,usable_chan_bool_test]) # USE ALL No uStim TRIALS
        z_Random = z_Random['z_mu']
        z_Random_targetdim = z_Random[0,target_dimension]
        
        # Update the prediction of MiSO online algorithm
        predictions[min_loss_stim_chan] = predictions[min_loss_stim_chan] - lr * (predictions[min_loss_stim_chan]-z_MiSO_targetdim)

        # Store log
        selected_chan_ind.append(min_loss_stim_chan)
        induced_activities_MiSO.append(z_MiSO_targetdim)
        induced_activities_noStim.append(z_noStim_targetdim)
        induced_activities_Random.append(z_Random_targetdim)
            
        # Choose the next pattern for MiSO with epsilon greedy algorithm
        l1_loss = np.abs(predictions - target).sum(axis=1)
        p = random.uniform(0, 1)
        if p<epsilon: # Exploration
            min_loss_stim_chan = np.random.choice(np.arange(0,96,1), size=1)[0]
        else: # Exploitation
            min_loss_stim_chan = np.argmin(l1_loss)
        
    performance_summary = {'induced_activities_MiSO': induced_activities_MiSO, 
                           'induced_activities_noStim': induced_activities_noStim,
                           'induced_activities_Random': induced_activities_Random,
                           'selected_chan_ind': selected_chan_ind,
                           'target': target}
    
    return performance_summary


def visualize_performnace(config_path_closed_loop, performance_summary, smoothing_window=10):
    """
    Visualize a closed-loop experiment performance.

    Parameters:
        config_path_closed_loop (str): Path to the closed-loop configuration file.
        performance_summary (dict): Closed-loop experiment performance summary.
    """
    # Load config
    with open(config_path_closed_loop, 'r') as f:
        config_closed_loop = yaml.safe_load(f)
    output_path = config_closed_loop['output'].get('path', '../')
    output_filename = config_closed_loop['output'].get('filename', 'performance_summary.png')
        
    # Load performance summary data for each method
    induced_activities_MiSO = performance_summary['induced_activities_MiSO']
    induced_activities_noStim = performance_summary['induced_activities_noStim']
    induced_activities_Random = performance_summary['induced_activities_Random']
    selected_chan_ind = performance_summary['selected_chan_ind']
    target = performance_summary['target']

    # Apply smoothing over trials
    induced_activities_MiSO_df = pd.DataFrame(induced_activities_MiSO)
    induced_activities_MiSO_moving_avg = induced_activities_MiSO_df.rolling(smoothing_window)
    induced_activities_MiSO_moving_avg = induced_activities_MiSO_moving_avg.mean()

    induced_activities_noStim_df = pd.DataFrame(induced_activities_noStim)
    induced_activities_noStim_moving_avg = induced_activities_noStim_df.rolling(smoothing_window)
    induced_activities_noStim_moving_avg = induced_activities_noStim_moving_avg.mean()

    induced_activities_Random_df = pd.DataFrame(induced_activities_Random)
    induced_activities_Random_moving_avg = induced_activities_Random_df.rolling(smoothing_window)
    induced_activities_Random_moving_avg = induced_activities_Random_moving_avg.mean()

    x = list(range(len(induced_activities_MiSO)))

    # Plot induced latent activity along target dimension 1
    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.plot(x, induced_activities_noStim_moving_avg.iloc[:,0], label='No uStim', color='lightgray', linewidth=1)
    plt.plot(x, induced_activities_Random_moving_avg.iloc[:,0], label='Random uStim', color='green', linewidth=1)
    plt.plot(x, induced_activities_MiSO_moving_avg.iloc[:,0], label='MiSO with single elec., CNN', color='deepskyblue', linewidth=1)
    plt.ylabel('Target dim 1', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(target[0], c='k', label='Target dim1', linewidth=1, linestyle='--')
    plt.legend(fontsize=10, ncol=4, bbox_to_anchor=(0.5, 1.05), loc='upper center')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Plot induced latent activity along target dimension 2
    plt.subplot(3,1,2)
    plt.plot(x, induced_activities_noStim_moving_avg.iloc[:,1], label='No uStim', color='lightgray', linewidth=1)
    plt.plot(x, induced_activities_Random_moving_avg.iloc[:,1], label='Random uStim', color='green', linewidth=1)
    plt.plot(x, induced_activities_MiSO_moving_avg.iloc[:,1], label='MiSO with single elec., CNN', color='deepskyblue', linewidth=1)
    plt.ylabel('Target dim 2', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(target[1], c='k', label='Target dim2', linewidth=1, linestyle='--')
    plt.legend(fontsize=10, ncol=4, bbox_to_anchor=(0.5, 1.05), loc='upper center')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Plot the selected electrode by MiSO
    plt.subplot(3,1,3)
    plt.scatter(x, np.array(selected_chan_ind)+1, s=2, color='deepskyblue', marker='|')
    plt.xlabel('Trial number in session', fontsize=15)
    plt.ylabel('Tested electrode', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False) 
    plt.ylim([1,96])
    plt.tight_layout()
    
    print(f"Saving figure to {output_path}{output_filename}")
    plt.savefig(f"{output_path}{output_filename}")
