#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 22:02:09 2023

@author: yang.kang
"""

from tqdm import tqdm
import os.path
import torch
import h5py
import numpy as np
from pathlib import Path
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device  = 'cpu'
import functions as fs
import PRIMME

### generate dataset

#trainset = fs.create_SPPARKS_dataset(size=[257,257], ngrains_rng=[256, 256], kt=0.66, cutoff=0.0, nsets=360, max_steps=500, offset_steps=301, future_steps=10)

### init parameters

paras_dict = {"num_eps": 25, 
              "mode": "Multi_Step", 
              "dims": 2, 
              "obs_dim":17, 
              "act_dim":17, 
              "lr": 5e-5, 
              "reg": 1, 
              "pad_mode": "circular",
              "pad_value": -1, 
              "if_plot": True}

trainset_dict = {"case1": ["./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_kt(0.66)_cut(0).h5"]}

n_samples_dict = {"case1": [[200]]}

'''
n_step_dict = {"case1": [["even_mixed", 1], ["even_mixed", 2], ["even_mixed", 3], ["even_mixed", 4], ["even_mixed", 5], ["even_mixed", 6], ["even_mixed", 7],
                         ["even_mixed", 8], ["even_mixed", 9], ["even_mixed", 10], ["even_mixed", 12], ["even_mixed", 14], ["even_mixed", 16]],
              }
'''

n_step_dict = {"case1": [[4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
                         [4, 8], [4, 9], [4, 10], [4, 12], [4, 14], [4, 16]],
              }

test_case_dict = {"case1": ["circular", [[257, 257], 64]],
                  "case2": ["circular", [[512, 512], 200]], 
                  "case3": ["square", [[512, 512], 64]],
                  "case4": ["square", [[512, 512], 200]],
                  "case5": ["hex", [[512, 512], 64]],
                  "case6": ["grain", [[512, 512], 512]],
                  "case7": ["grain", [[1024, 1024], 2**12]]} # "case8": ["grain", [[2400, 2400], 24000]]

### Train PRIMME using the above training set from SPPARKS

for key_data in trainset_dict.keys():
    trainset = trainset_dict[key_data]
    for n_samples in n_samples_dict[key_data]:
        for n_step in n_step_dict[key_data]:
            PRIMME.train_primme(trainset, n_step, n_samples, test_case_dict)

'''-------------------------------------------------------------------------'''

for key_data in trainset_dict.keys():
    trainset = trainset_dict[key_data]
    for n_samples in n_samples_dict[key_data]:
        for n_step in n_step_dict[key_data]:
            dataset_name = trainset[0].split("spparks_")[1].split("_kt")[0]
            subfolder = "pred(%s)_%s_step(%s_%s)_ep(%d)_pad(%s)_md(%d)_sz(%d_%d)_lr(%.0e)_reg(%s)" % (paras_dict['mode'], dataset_name, n_step[0], n_step[1], 25, paras_dict['pad_mode'], 
                                                                                                      paras_dict['dims'], paras_dict['obs_dim'], paras_dict['act_dim'], 
                                                                                                      paras_dict['lr'], paras_dict['reg'])
            modelname = './data/' + subfolder + '.h5'
            nsteps = 1800
            for key in test_case_dict.keys():
                grain_shape, grain_sizes = test_case_dict[key]
                if grain_shape == "hex":
                    ic_shape = grain_shape
                else:   
                    ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
                filename_test = ic_shape + ".pickle"    
                path_load = Path('./data').joinpath(filename_test)
                if os.path.isfile(str(path_load)):  
                    data_dict = fs.load_picke_files(load_dir = Path('./data'), filename_save = filename_test)
                    ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]  
                    print("1", str(path_load))
                else:
                    ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
                    print("2", str(path_load))
                ## Run PRIMME model
                ims_id, fp_primme = PRIMME.run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname)
                sub_folder = "pred" + fp_primme.split("/")[2].split(".")[0].split("pred")[1]
                fs.compute_grain_stats(fp_primme)
                fs.make_videos(fp_primme, sub_folder, ic_shape)
                if grain_shape == "grain":
                    fs.make_time_plots(fp_primme, sub_folder, ic_shape)            
            


'''-------------------------------------------------------------------------'''

### Run SPPARKS model
sub_folder = "SPPARK"
for key in reversed(test_case_dict.keys()):
    grain_shape, grain_sizes = test_case_dict[key]
    if grain_sizes[0][0] > 1000:
        nsteps = 1300;
    else:
        nsteps = 1800
    if grain_shape == "hex":
        ic_shape = grain_shape
    else:   
        ic_shape = grain_shape + "(" + ("_").join([str(grain_sizes[0][0]), str(grain_sizes[0][1]), str(grain_sizes[1])]) + ")"
    filename_test = ic_shape + ".pickle"    
    path_load = Path('./data').joinpath(filename_test)
    if os.path.isfile(str(path_load)):  
        data_dict = fs.load_picke_files(load_dir = Path('./data'), filename_save = filename_test)
        ic, ea, miso_array, miso_matrix = data_dict["ic"], data_dict["ea"], data_dict["miso_array"], data_dict["miso_matrix"]  
        print("1", str(path_load))
    else:
        ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
        print("2", str(path_load))
    fp_save = './data/spparks_ic_shape(%s)_nsteps(%d)_kt(%.2f)_cut(%d).h5'%(ic_shape, nsteps, 0.66, 0.0)    
    ims_id, fp_spparks = fs.run_spparks(ic, ea, nsteps=nsteps, kt=0.66, cut=0.0, fp_save = fp_save) # 1000
    fs.compute_grain_stats(fp_spparks) # gps='sim0'
    fs.make_videos(fp_spparks, sub_folder, ic_shape) #saves to 'plots'
    if grain_shape == "grain":
        fs.make_time_plots(fp_spparks, sub_folder, ic_shape) #saves to 'plots'


'''-------------------------------------------------------------------------'''


data_dict = {}
for ic_shape in ["circular(257_257_64)"]:
    
    data_dict[ic_shape] = {}
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "/home/UFAD/yang.kang/Ultrasonics/Kang/GrainGrowth/primme_share/PyPRIMME_Multi_Step_case_03/data/spparks_ic_shape(%s)_nsteps(1300)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "/home/UFAD/yang.kang/Ultrasonics/Kang/GrainGrowth/primme_share/PyPRIMME_Multi_Step_case_03/data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
        
    hps = [spparks_sim,
           "/home/UFAD/yang.kang/Ultrasonics/Kang/GrainGrowth/primme_share/PyPRIMME_Single_Step/data/primme_shape(%s)_pred(Single_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_1)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_4)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_6)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_8)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_10)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_12)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_14)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_16)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
          ]

    gps='sim0'     
    if type(hps)!=list: hps = [hps]
    if type(gps)!=list: gps = [gps]   
    
    for i in range(len(hps)):
        if "spparks" in hps[i]:
            key_data = "spparks"
        elif "Single_Step" in hps[i]:
            key_data = "PRIMME"
        else:
            key_data = hps[i].split("max(100)_")[1].split("_ep(25)")[0]
        
        data_dict[ic_shape][key_data] = {}
        with h5py.File(hps[i], 'r') as f: 
            print(hps[i])
            print(f.keys())
            print(f[gps[0]].keys())
            #ims_id_area = np.sum(np.sum(f[gps[0]]['ims_id'][:, 0, :, :], axis = 2), axis = 1)
            data_dict[ic_shape][key_data]['ims_id'] = f[gps[0]]['ims_id'][:, 0,]
            data_dict[ic_shape][key_data]['grain_areas'] = f[gps[0]]['grain_areas'][:]
            data_dict[ic_shape][key_data]['grain_sides'] = f[gps[0]]['grain_sides'][:]
            data_dict[ic_shape][key_data]['grain_areas_avg'] = f[gps[0]]['grain_areas_avg'][:]
            data_dict[ic_shape][key_data]['grain_sides_avg'] = f[gps[0]]['grain_sides_avg'][:]
'''    
hps = [spparks_sim,
       "/home/UFAD/yang.kang/Ultrasonics/Kang/GrainGrowth/primme_share/PyPRIMME_Single_Step/data/primme_shape(%s)_pred(Single_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_1)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_4)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_6)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_8)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_10)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_12)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_14)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
       "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(even_mixed_16)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
      ]    

'''    
    
    
    
sub_folder = "paper2"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)          







