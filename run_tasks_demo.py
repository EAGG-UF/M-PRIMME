#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:35:33 2023

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



test_case_dict = {"case1": ["circular", [[257, 257], 64]],
                  "case2": ["circular", [[512, 512], 200]], 
                  "case3": ["square", [[512, 512], 64]],
                  "case4": ["square", [[512, 512], 200]],
                  "case5": ["hex", [[512, 512], 64]],
                  "case6": ["grain", [[512, 512], 512]],
                  "case7": ["grain", [[1024, 1024], 2**12]]} # "case8": ["grain", [[2400, 2400], 24000]]

modelname = "./data/pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"

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
    else:
        ic, ea, miso_array, miso_matrix = fs.generate_train_init(filename_test, grain_shape, grain_sizes, device)
    ## Run PRIMME model
    ims_id, fp_primme = PRIMME.run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname)
    sub_folder = "pred" + fp_primme.split("/")[2].split(".")[0].split("pred")[1]
    fs.compute_grain_stats(fp_primme)
    fs.make_videos(fp_primme, sub_folder, ic_shape)
    if grain_shape == "grain":
        fs.make_time_plots(fp_primme, sub_folder, ic_shape)