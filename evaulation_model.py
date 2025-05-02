#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:01:16 2023

@author: yang.kang
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import functions as fs
import visualization

'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path) 
# for ic_shape in ["grain(512_512_512)", "grain(1024_1024_4096)"]:
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "primme_shape(%s)_step(1_1)_varying_epoch"%ic_shape
    legend = ["SPPARKS"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = [spparks_sim,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(5)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(10)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(15)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(20)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    for i in range(1, len(hps)):
        legend.append(hps[i].split("/")[2].split("_pad")[0].split("pred(Multi_Step)_")[1].replace("_nsets(200)_future(10)_max(100", ""))
    if "grain" in ic_shape: 
        visualization.make_time_plots(hps, sub_folder, figname, ic_shape, legend = legend)
    else:
        visualization.make_time_plots_simple_grain(hps, figname, sub_folder, ic_shape, legend = legend)

'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)  
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "primme_shape(%s)_step(2_2)_varying_epoch"%ic_shape
    legend = ["SPPARKS"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = [spparks_sim,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(5)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(5)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(10)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(15)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(20)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    for i in range(1, len(hps)):
        legend.append(hps[i].split("/")[2].split("_pad")[0].split("pred(Multi_Step)_")[1].replace("_nsets(200)_future(10)_max(100", ""))
    if "grain" in ic_shape: 
        visualization.make_time_plots(hps, sub_folder, figname, ic_shape, legend = legend)
    else:
        visualization.make_time_plots_simple_grain(hps, figname, sub_folder, ic_shape, legend = legend)


'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)  
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "primme_shape(%s)_step(5_5)_varying_epoch"%ic_shape
    legend = ["SPPARKS"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = [spparks_sim,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(5)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(5)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(10)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(15)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(20)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    for i in range(1, len(hps)):
        legend.append(hps[i].split("/")[2].split("_pad")[0].split("pred(Multi_Step)_")[1].replace("_nsets(200)_future(10)_max(100", ""))
    if "grain" in ic_shape: 
        visualization.make_time_plots(hps, sub_folder, figname, ic_shape, legend = legend)
    else:
        visualization.make_time_plots_simple_grain(hps, figname, sub_folder, ic_shape, legend = legend)

    
'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)  
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "primme_shape(%s)_step(1_8_0)_varying_epoch"%ic_shape
    legend = ["SPPARKS"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = [spparks_sim,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(1_1)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(2_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(3_3)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(4_4)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(5_5)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(6_6)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(7_7)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_8)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    for i in range(1, len(hps)):
        legend.append(hps[i].split("/")[2].split("_pad")[0].split("pred(Multi_Step)_")[1].replace("_nsets(200)_future(10)_max(100", ""))
    if "grain" in ic_shape: 
        visualization.make_time_plots(hps, sub_folder, figname, ic_shape, legend = legend)
    else:
        visualization.make_time_plots_simple_grain(hps, figname, sub_folder, ic_shape, legend = legend)
  
    
'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)  
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "primme_shape(%s)_step(8_8)_varying_step"%ic_shape
    legend = ["SPPARKS"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = [spparks_sim,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_4)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_6)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_8)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_10)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_12)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_14)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    for i in range(1, len(hps)):
        legend.append(hps[i].split("/")[2].split("_pad")[0].split("pred(Multi_Step)_")[1].replace("_nsets(200)_future(10)_max(100", ""))
    if "grain" in ic_shape: 
        visualization.make_time_plots(hps, sub_folder, figname, ic_shape, legend = legend)
    else:
        visualization.make_time_plots_simple_grain(hps, figname, sub_folder, ic_shape, legend = legend)

'''-------------------------------------------------------------------------'''



sub_folder = "time_plots_simple_grain"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)  
for ic_shape in ["circular(257_257_64)", "circular(512_512_200)", "square(512_512_64)", "square(512_512_200)", "grain(512_512_512)", "grain(1024_1024_4096)"]:
    figname = "merge_primme_shape(%s)_step(8_8)_varying_step"%ic_shape
    legend = ["Average of steps from 8-2 to 8-12"]
    if ic_shape in ["grain(1024_1024_4096)"]:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1400)_kt(0.66)_cut(0).h5"%ic_shape
    else:
        spparks_sim = "./data/spparks_ic_shape(%s)_nsteps(1800)_kt(0.66)_cut(0).h5"%ic_shape
    hps = ["./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_2)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_4)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_6)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_8)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_10)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_12)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape,
           "./data/primme_shape(%s)_pred(Multi_Step)_sz(257x257)_ng(256-256)_nsets(200)_future(10)_max(100)_step(8_14)_ep(25)_pad(circular)_md(2)_sz(17_17)_lr(5e-05)_reg(1).h5"%ic_shape]
    visualization.make_merge_videos(hps, sub_folder, figname, ic_shape, legend = legend)

    
       
    
    
    
'''-------------------------------------------------------------------------'''

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))


filename_hdf = "./data/s1400poly1_t0_optimized.dream3d"
filename_hdf = "./data/s1400poly1_t1_fiftytwoslices_optimized.dream3d"
with h5py.File(filename_hdf, 'r') as hf:
    print(hf)
    h5_tree(hf)
    
data = h5py.File(filename_hdf, 'r') 

data["DataContainers"]["ImageDataContainer"]["CellData"]["FeatureIds"]

data["DataContainers"]["ImageDataContainer"]["CellData"]["X Position"].shape


data["DataContainers"]["ImageDataContainer"]["CellFeatureData"]["Centroids"][1:]


    
'''-------------------------------------------------------------------------'''

sub_folder = "time_plots_experiment"
result_path = ("/").join(['./plots', sub_folder])
if not os.path.exists(result_path):
   os.makedirs(result_path)   
for n in range(50):
    plt.figure(figsize = (10, 10))
    plt.imshow(data["DataContainers"]["ImageDataContainer"]["CellData"]["FeatureIds"][n, :, :, 0])
    plt.savefig('./plots/%s/s1400poly1_t0_optimized(%s)'%(sub_folder, "s1400poly1_t0_optimized(%s)"%n), dpi=300)
    plt.close()




filename_hdf = "./data/s1400poly1_t1_fiftytwoslices_optimized.dream3d"
data_2 = h5py.File(filename_hdf, 'r') 


for n in range(52):
    plt.figure(figsize = (10, 10))
    plt.imshow(data_2["DataContainers"]["ImageDataContainer"]["CellData"]["FeatureIds"][n, :, :, 0])
    plt.savefig('./plots/%s/s1400poly1_t1_fiftytwoslices_optimized(%s)'%(sub_folder, "s1400poly1_t0_optimized(%s)"%n), dpi=300)
    plt.close()


data_2["DataContainers"]["ImageDataContainer"]["CellFeatureData"]["Centroids"][1:].shape



sgl = [] 
myfile = open('./data/sgl_1_09222021.txt', 'r')
for line in myfile:
    print(line)
    sgl.append(int(line))
sgl = np.array(sgl)

filename_hdf = "./data/s1400poly1_t0_optimized.dream3d"
data_0 = h5py.File(filename_hdf, 'r')
centroids_t0 = data_0["DataContainers"]["ImageDataContainer"]["CellFeatureData"]["Centroids"][:]
filename_hdf = "./data/s1400poly1_t1_fiftytwoslices_optimized.dream3d"
data_1 = h5py.File(filename_hdf, 'r')
centroids_t1 = data_1["DataContainers"]["ImageDataContainer"]["CellFeatureData"]["Centroids"][:]

xyz_diff = []
for i in range(len(sgl)-1):
    n = sgl[i]
    if n != 0:
        xyz = (centroids_t1[n] - centroids_t0[i+1])/2
        xyz_diff.append(xyz)
xyz_diff = np.array(xyz_diff)

#np.mean(xyz_diff, axis = 0)

xlabel = ["Centroid (x-shift)", "Centroid (y-shift)", "Centroid (z-shift)"]
plt.figure(figsize = (15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(xyz_diff[:, i], bins = 100)
    plt.xlabel(xlabel[i])
plt.show()



FeatureIds_0 = data_0["DataContainers"]["ImageDataContainer"]["CellData"]["FeatureIds"][:]
FeatureIds_1 = data_1["DataContainers"]["ImageDataContainer"]["CellData"]["FeatureIds"][:]

FeatureIds_0_shift = np.zeros(FeatureIds_1.shape[:3])
for i in range(FeatureIds_1.shape[0]):
    for j in range(FeatureIds_1.shape[1]):
        for k in range(FeatureIds_1.shape[1]):
            i_0 = max(i-6, 0)
            j_0 = max(j-2, 0)
            k_0 = max(k-3, 0)
            FeatureIds_0_shift[i, j, k] = FeatureIds_0[i_0, j_0, k_0, 0]
            

for n in range(50):
    plt.figure(figsize = (16, 5))
    plt.subplot(131)
    plt.imshow(FeatureIds_0[n, :, :, 0])
    plt.title("t0(%s)"%n)
    plt.subplot(132)
    plt.imshow(FeatureIds_0_shift[n])
    plt.title("t0(%s)-shift"%n)
    plt.subplot(133)
    plt.imshow(FeatureIds_1[n, :, :, 0])
    plt.title("t1(%s)"%n)
    plt.savefig('./plots/%s/t0_t1_grain(%s)'%(sub_folder, n), dpi=300)
    plt.close()


from scipy import ndimage

    
angle = 47
for n in range(50):
    plt.figure(figsize = (16, 10))
    plt.subplot(231)
    plt.imshow(FeatureIds_0[n, :, :, 0])
    plt.title("t0(%s)"%n)
    plt.subplot(232)
    plt.imshow(FeatureIds_0_shift[n])
    plt.title("t0(%s)-shift"%n)
    plt.subplot(233)
    plt.imshow(FeatureIds_1[n, :, :, 0])
    plt.title("t1(%s)"%n)
    plt.subplot(234)
    img_0_rot = ndimage.rotate(FeatureIds_0[n, :, :, 0], angle, reshape=False)
    plt.imshow(img_0_rot)
    plt.title("t0_rotation(%s)"%n)
    plt.subplot(235)
    img_0_shift_rot = ndimage.rotate(FeatureIds_0_shift[n], angle, reshape=False)
    plt.imshow(img_0_shift_rot)
    plt.title("t0_shift_rotation(%s)"%n)
    plt.subplot(236)
    img_1_rot = ndimage.rotate(FeatureIds_1[n, :, :, 0], angle, reshape=False)
    plt.imshow(img_1_rot)
    plt.title("t1_rotation(%s)"%n)    
    plt.savefig('./plots/%s/t0_t1_grain_rotation(%s)'%(sub_folder, n), dpi=300)
    plt.close()

    





