#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run MiSO using 96 single electrode uSgim patterns

Author: Yuki Minai
Version: 1.0.0
License: CC BY-NC-ND 4.0
"""
 
import numpy as np
import pandas as pd
import yaml

from spike_data_generator import generate_spike_count_data
from fa_trainer import fit_fa
from train_data_creator import create_training_data
from cnn_trainer import train_cnn
from closed_loop_experiment_executor import run_closed_loop_experiment, visualize_performnace


def main():
    print('Loading config paths')
    with open('../config/config_path.yaml', 'r') as f:
        config_path = yaml.safe_load(f)
    config_path_base = config_path['config_path_base']    
    config_spike_cnt_ref = config_path_base+config_path['spike_cnt_ref']
    config_spike_cnt_train = config_path_base+config_path['spike_cnt_train']
    config_spike_cnt_test = config_path_base+config_path['spike_cnt_test']
    config_fa = config_path_base+config_path['fa']
    config_cnn = config_path_base+config_path['cnn']
    config_closed_loop = config_path_base+config_path['closed_loop']
    
    print('Loading reference, train, and test session file names')
    with open(config_path_base+config_path['filenames'], 'r') as f:
        config_filenames = yaml.safe_load(f) 
    filename_ref = config_filenames['referenfce']
    filenames_train = config_filenames['train']
    filename_test = config_filenames['test']
        
    print('\n****************')
    print('Step1. Run a reference session which involves only non-uStim trials')
    print(config_spike_cnt_ref)
    generate_spike_count_data(filename_ref, config_spike_cnt_ref)
    
    print('\n****************')
    print('Step2. Fit FA using a reference session data')
    fit_fa(filename_ref, config_fa)
    
    print('\n****************')
    print('Step3. Run CNN training data collection sessions')
    generate_spike_count_data(filenames_train, config_spike_cnt_train)

    print('\n****************')
    print('Step4. Fit FA using training data')
    fit_fa(filenames_train, config_fa)
    
    print('\n****************')
    print('Step5. Align training FA loadings to a reference loading and create CNN training data')
    create_training_data(filename_ref, filenames_train)
    
    print('\n****************')
    print('Step6. Train CNN and obtain predictions')
    train_cnn(config_cnn)
    
    print('\n****************')
    print('Step7. Run a closed-loop optimization session')
    performance_summary = run_closed_loop_experiment(config_closed_loop, config_spike_cnt_test, config_fa, filename_ref, filename_test)
    
    print('\n****************')
    print('Step8. Visualize a closed-loop performance')
    visualize_performnace(config_closed_loop, performance_summary)
    
    
if __name__=="__main__":
    main()