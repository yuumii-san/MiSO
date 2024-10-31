#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a training dataset for CNN model by using latent space alignment.

This script aligns FA subspace of each session to a reference FA subspace
using FA+Procrustes method and creates CNN training data.

Author: Yuki Minai
Created: 2024-11-01
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""

import sys
import copy
import numpy as np
import pandas as pd
import pickle

sys.path.append("./util/")
from fa import factor_analysis as fa_mdl
from fa_loading_aligner import align_loadings


def create_training_data(filename_ref, filename_train):
    """
    Create CNN training data by using latent space alignment.

    Parameters:
        filename_ref (str): Filename of a reference session.
        filename_train (list of str): List of filenames of training sessions.

    Returns:
        list: CNN training input. (stimulated electrode)
        list: CNN training output. (induced latent response)
    """
        
    # Load FA model for a reference session
    fa_file_path_ref = f"../model/fa/fa_{filename_ref[0]}.pkl"
    with open(fa_file_path_ref, 'rb') as f:
        fa_model_ref = pickle.load(f)
        
    # Load usable channel for a reference session
    usable_chan_list_path_ref = f"../data/usable_channels/usable_channel_list_{filename_ref[0]}.csv"
    usable_chan_bool_path_ref = f"../data/usable_channels/usable_channel_bool_{filename_ref[0]}.csv"
    usable_chan_ref = pd.read_csv(usable_chan_list_path_ref, header=None)
    usable_chan_ref = list(usable_chan_ref[0])  # len = number of usable chan
    usable_chan_bool_ref = pd.read_csv(usable_chan_bool_path_ref, header=None)
    usable_chan_bool_ref = list(usable_chan_bool_ref[0])  # len = 96
    
    # Load FA model, usable channel list, binned spike data for training sessions
    fa_models, usable_channels, usable_channels_bool, binned_spikes, stim_chans = [], [], [], [], []
    for filename in filename_train:
        # Load FA model
        fa_file_path = f"../model/fa/fa_{filename}.pkl"
        with open(fa_file_path, 'rb') as f:
            fa_model = pickle.load(f)
            fa_models.append(fa_model)

        # Load usable channel list
        usable_chan_list_path = f"../data/usable_channels/usable_channel_list_{filename}.csv"
        usable_chan = pd.read_csv(usable_chan_list_path, header=None)
        usable_chan = list(usable_chan[0])  # len = number of usable chan
        usable_channels.append(usable_chan)
        usable_chan_bool_path = f"../data/usable_channels/usable_channel_bool_{filename}.csv"
        usable_chan_bool = pd.read_csv(usable_chan_bool_path, header=None)
        usable_chan_bool = list(usable_chan_bool[0])  # len = 96
        usable_channels_bool.append(usable_chan_bool)
        
        # Load binned spike data
        binned_spike_file_path = f"../output/binned_spike_cnt/binned_spike_cnt_{filename}.csv"
        binned_spike = pd.read_csv(binned_spike_file_path)
        binned_spikes.append(np.array(binned_spike.iloc[:, :-1]))  # drop stim_chan column
        stim_chans.append(np.array(binned_spike.iloc[:, -1]))  # extract stim_chan column

    # Align loadings to a reference session
    aligned_loadings, aligned_zs, fa_model_aligneds, stim_chans_aligneds = [], [], [], []
    for i, filename in enumerate(filename_train):
        # Align loadings (1: reference session, 2: align target session)
        keep_chan1, keep_chan2 = usable_chan_ref, usable_channels[i]
        L1, L2 = fa_model_ref.fa_params['L'], fa_models[i].fa_params['L']
        aligned_L2 = align_loadings(L1, L2, keep_chan1, keep_chan2)    
        aligned_loadings.append(aligned_L2)
        
        # Update fa model loading        
        fa_model_aligned = copy.deepcopy(fa_models[i])
        fa_model_aligned.fa_params['L'] = aligned_L2
        
        # Compute posterior with aligned loading
        z, LL_fit = fa_model_aligned.estep(binned_spikes[i][:, usable_channels_bool[i]])
        aligned_zs.append(z['z_mu'])
        fa_model_aligneds.append(fa_model_aligned)
        stim_chans_aligneds.append(list(stim_chans[i]))
            
    aligned_zs = np.vstack(aligned_zs)
    stim_chans_aligneds = [chan for chans in stim_chans_aligneds for chan in chans]

    # Save training data
    print("Saving aligned latent data for CNN training to ../output/cnn_training_data/cnn_training_data_zs.csv")
    np.savetxt("../output/cnn_training_data/cnn_training_data_zs.csv", aligned_zs, delimiter=',', fmt='%s') 
    print("Saving uStim pattern data for CNN training to ../output/cnn_training_data/cnn_training_data_stim_chans.csv")
    np.savetxt("../output/cnn_training_data/cnn_training_data_stim_chans.csv", stim_chans_aligneds, delimiter=',', fmt='%s') 

    return stim_chans_aligneds, aligned_zs
