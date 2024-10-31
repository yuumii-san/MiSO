#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sim_fa as sf
import factor_analysis as fa_mdl
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# simulate from a factor analysis model
fa_simulator = sf.sim_fa(20,5,model_type='fa',rand_seed=0)
X = fa_simulator.sim_data(10000,rand_seed=0)
sim_params = fa_simulator.get_params()

# # fit fa model
# model = fa_mdl.factor_analysis(model_type='fa')
# log_L = model.train(X,5,verbose=False,rand_seed=0)
# fit_params = model.get_params()

# cross-validation

model = fa_mdl.factor_analysis(model_type='fa')
start = timer()
LL_curve1 = model.crossvalidate(X,rand_seed=0)
end = timer()
serial_time = end-start
model = fa_mdl.factor_analysis(model_type='fa')
start = timer()
LL_curve2 = model.crossvalidate(X,parallelize=True,rand_seed=0)
end = timer()
par_time = end-start

# plot cross-validation curve
plt.figure(0)
plt.plot(LL_curve1['z_list'],LL_curve1['LLs'],'bo-')
plt.plot(LL_curve1['zDim'],LL_curve1['final_LL'],'r^')
plt.show()
plt.figure(1)
plt.plot(LL_curve2['z_list'],LL_curve2['LLs'],'bo-')
plt.plot(LL_curve2['zDim'],LL_curve2['final_LL'],'r^')
plt.show()

print(LL_curve1['LLs']-LL_curve2['LLs'])
print('parallel: ',par_time,' sec')
print('serial: ',serial_time,' sec')

## get latents and compare recovered latents vs true latents
#z_fit,LL_fit = model.estep(X)
#z_fit,Lorth = model.orthogonalize(z_fit['z_mu'])
#
#sim_model = fa_mdl.factor_analysis(model_type='fa')
#sim_model.set_params(sim_params)
#z_true,LL_true = sim_model.estep(X)
#z_true,Lorth = sim_model.orthogonalize(z_true['z_mu'])
#
#plt.figure(1)
#plt.plot(z_true[:,0],z_fit[:,0],'b.')
#plt.xlabel('True z1')
#plt.ylabel('Recovered z1')
#plt.show()


## compute fa metrics
#fitted_metrics = model.compute_metrics(cutoff_thresh=0.95)
#true_metrics = sim_model.compute_metrics(cutoff_thresh=0.95)
#
#print('fitted psv:',fitted_metrics['psv'])
#print('ground truth psv:',true_metrics['psv'])
#print('fitted dshared:',fitted_metrics['dshared'])
#print('ground truth dshared:',true_metrics['dshared'])
#print('fitted participation ratio: {:.2f}'.format(fitted_metrics['part_ratio']))
#print('ground truth participation ratio: {:.2f}'.format(true_metrics['part_ratio']))
