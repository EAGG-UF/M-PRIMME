#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script is used to train the PRIMME model using data from SPPARKS, generated in real time.
    A SPPARKS environment must be set up for this script to run properly. Please see ./spparks_files/Getting_Started
    The model is saved to "./saved models/" after each training epoch
    Training evaluation files are saved to "./results_training" after training is complete

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite (https://arxiv.org/abs/2203.03735): 
    Yan, Weishi, et al. "Predicting 2D Normal Grain Growth using a Physics-Regularized Interpretable Machine Learning Model." arXiv preprint arXiv:2203.03735 (2022).
"""

# IMPORT PACKAGES
from tqdm import tqdm
import os.path
import torch
import h5py
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device  = 'cpu'
import functions as fs
import PRIMME

### Create training set by running SPPARKS
# trainset = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=1, max_steps=1000, offset_steps=1, future_steps=4)
# trainset = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=1, max_steps=202, offset_steps=0, future_steps=202)


### Train PRIMME using the above training set from SPPARKS
trainset= ["./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(200)_max(200)_kt(0.66)_cut(0).h5",
           "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(202)_max(202)_kt(0.66)_cut(0).h5",
           "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(1000)_max(1000)_kt(0.66)_cut(0).h5",
           "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(1001)_max(1001)_kt(0.66)_cut(0).h5"]
trainset= ["./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(200)_max(200)_kt(0.66)_cut(0).h5"]
trainset= ["./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(1)_future(1000)_max(1000)_kt(0.66)_cut(0).h5"]
# model_location = PRIMME.train_primme(trainset, offsets = [[0, 200]], num_eps=20, mode = "Multi_Step", dims=2, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", pad_value = -1, if_plot=True)

### VALIDATION

## Choose initial conditions
ic, ea = fs.generate_circleIC(size=[257,257], r=64) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_circleIC(size=[512,512], r=200) #nsteps=200, pad_mode='circular'
# ic, ea = fs.generate_3grainIC(size=[512,512], h=350) #nsteps=300, pad_mode=['reflect', 'circular']
# ic, ea = fs.generate_hexIC() #nsteps=500, pad_mode='circular'
# ic, ea = fs.generate_SquareIC(size=[512,512], r=200) 
# ic, ea, _ = fs.voronoi2image(size=[257, 257], ngrain=256) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[1024, 1024], ngrain=2**12) #nsteps=1000, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2048, 2048], ngrain=2**14) #nsteps=1500, pad_mode='circular'
# ic, ea, _ = fs.voronoi2image(size=[2400, 2400], ngrain=24000, device = "cpu") #nsteps=1500, pad_mode='circular'

## Run PRIMME model
for Nseq in [7, 11, 16, 21, 32, 40, 42, 46, 50, 52, 62, 99, 124, 199]:
    modelname = './data/pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_offset(0-200)_sz(257x257)_max(200).h5'%Nseq
    #modelname = './data/pred(Multi_Step)_pad(circular)_Nseq(23)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(20)_offset(0-200)_sz(257x257)_max(200)_offset(0-200)_sz(257x257)_max(202)_offset(0-200)_sz(257x257)_max(1000).h5'
    ic_shape = "grain(1024-4096)" # 
    ims_id, fp_primme = PRIMME.run_primme(ic=ic, ea=ea, ic_shape=ic_shape, nsteps=1800, n_samples=Nseq, reg = 1, modelname=modelname, pad_mode="circular", 
                                          pad_value=-1, mode="Multi_Step", if_plot=True)
    fs.compute_grain_stats(fp_primme) # 
    sub_folder = "pred(Multi_Step)_epoch(0)_Nseq(%s)"%Nseq
    fs.make_videos(fp_primme, sub_folder) #saves to 'plots'
    fs.make_time_plots(fp_primme, sub_folder) #saves to 'plots'

## Run SPPARKS model
# ims_id, fp_spparks = fs.run_spparks(ic, ea, nsteps=1400, kt=0.66, cut=0.0) # 1000
# fs.compute_grain_stats(fp_spparks, gps='sim1') # gps='sim0'
# sub_folder = "SPPARK"
# fs.make_videos(fp_spparks, sub_folder) #saves to 'plots'
# fs.make_time_plots(fp_spparks, sub_folder) #saves to 'plots'

Nseq = 49
for offset in [[200, 400], [300, 500], [400, 600], [500, 700], [600, 800]]:
    modelname = './data/pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_offset(%s-%s)_sz(257x257)_max(1000).h5'% (Nseq, offset[0], offset[1])
    ic_shape = "grain(1024-4096)" # 
    ims_id, fp_primme = PRIMME.run_primme(ic=ic, ea=ea, ic_shape=ic_shape, nsteps=1800, n_samples=Nseq, reg = 1, modelname=modelname, pad_mode="circular", 
                                          pad_value=-1, mode="Multi_Step", if_plot=True)
    fs.compute_grain_stats(fp_primme) # 
    sub_folder = "pred(Multi_Step)_epoch(0)_Nseq(%s)"%Nseq
    fs.make_videos(fp_primme, sub_folder) #saves to 'plots'
    fs.make_time_plots(fp_primme, sub_folder) #saves to 'plots'

for Nseq in [23, 28, 39, 48]:
    #modelname = './data/pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_offset(0-200)_sz(257x257)_max(200).h5'%Nseq
    modelname = './data/pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(20)_offset(0-200)_sz(257x257)_max(200)_offset(0-200)_sz(257x257)_max(202)_offset(0-200)_sz(257x257)_max(1000).h5'%Nseq
    ic_shape = "grain(1024-4096)_mixed_3" # 
    ims_id, fp_primme = PRIMME.run_primme(ic=ic, ea=ea, ic_shape=ic_shape, nsteps=1800, n_samples=Nseq, reg = 1, modelname=modelname, pad_mode="circular", 
                                          pad_value=-1, mode="Multi_Step", if_plot=True)
    fs.compute_grain_stats(fp_primme) # 
    sub_folder = "pred(Multi_Step)_epoch(0)_Nseq(%s)"%Nseq
    fs.make_videos(fp_primme, sub_folder) #saves to 'plots'
    fs.make_time_plots(fp_primme, sub_folder) #saves to 'plots'

'''-------------------------------------------------------------------------'''
## Compare PRIMME and SPPARKS statistics
fp_spparks = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1400)_freq(1)_kt(0.66)_cut(0).h5'
for Nseq in [7, 11, 16, 21, 32, 40, 42, 46, 50, 52, 62, 99, 124, 199]:
    ic_shape = "grain(1024-4096)"
    fp_primme = './data/primme_shape(%s)_pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_offset(0-200)_sz(257x257)_max(200).h5'%(ic_shape, Nseq)
    sub_folder = "SPPARK" # "pred(Multi_Step)_epoch(0)_Nseq(%s)"%Nseq
    legend = ["SPPAKS", "PRIMME(Multi_Step)_Nseq(%s)_offset(0-200)"%(Nseq)]
    hps = [fp_spparks, fp_primme]
    fs.make_time_plots(hps, sub_folder, legend)
   
Nseq = 49
fp_spparks = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1400)_freq(1)_kt(0.66)_cut(0).h5'
for offset in [[200, 400], [300, 500], [400, 600], [500, 700], [600, 800]]:
    ic_shape = "grain(1024-4096)" # 
    fp_primme = './data/primme_shape(%s)_pred(Multi_Step)_pad(circular)_Nseq(49)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(10)_offset(%s-%s)_sz(257x257)_max(1000).h5'%(ic_shape, offset[0], offset[1])
    sub_folder = "SPPARK" #  sub_folder = "pred(Multi_Step)_epoch(0)_Nseq(%s)"%Nseq
    legend = ["SPPAKS", "PRIMME(Multi_Step)_Nseq(49)_offset(%s-%s)"%(offset[0], offset[1])]
    hps = [fp_spparks, fp_primme]
    fs.make_time_plots(hps, sub_folder, legend) #saves to 'plots'
 
fp_spparks = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1400)_freq(1)_kt(0.66)_cut(0).h5'
for Nseq in [23, 28, 39, 48]:
    ic_shape = "grain(1024-4096)_mixed_3" #
    fp_primme = './data/primme_shape(grain(1024-4096)_mixed_3)_pred(Multi_Step)_pad(circular)_Nseq(%s)_model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(20)_offset(0-200)_sz(257x257)_max(200)_offset(0-200)_sz(257x257)_max(202)_offset(0-200)_sz(257x257)_max(1000).h5'%Nseq
    sub_folder = "SPPARK"
    legend = ["SPPAKS", "PRIMME(Multi_Step)_Nseq(%s)_offset(0-200)"%(Nseq)] # 
    hps = [fp_spparks, fp_primme]
    fs.make_time_plots(hps, sub_folder, legend) #saves to 'plots'

   
'''-------------------------------------------------------------------------'''


import matplotlib.pyplot as plt

plt.figure()
plt.imshow(ic)
plt.show()

im_seq = []
plt.figure(figsize = (18, 5))
for n, i in enumerate([0, 2, 4]):
    plt.subplot(1, 3, n+1)
    plt.imshow(im_seq[0, 20 * i, 0])
plt.show()   

 
import pickle

data_dict = {'ic':ic, 'ea':ea}
with open('./data/ic(1024x1024).pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
with open('./data/ic(1024x1024).pickle', 'rb') as handle:
    data_dict = pickle.load(handle)

ic = data_dict['ic']
ea = data_dict['ea']

print(all(data_dict == data_dict))   
    
