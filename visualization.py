#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 01:15:35 2023

@author: yang.kang
"""
import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
c = [mcolors.TABLEAU_COLORS[n] for n in list(mcolors.TABLEAU_COLORS)]
import imageio
from sklearn.linear_model import LinearRegression
from pathlib import Path

def make_time_plots_simple_grain(hps, figname, sub_folder,  ic_shape, gps='sim0', legend = []):
        
    if type(hps)!=list: hps = [hps]
    if type(gps)!=list: gps = [gps]
    
    N = 1200
    plt.figure()
    for i in tqdm(range(len(hps)),'Calculating grain area'):
        with h5py.File(hps[i], 'r') as f: 
            #ims_id_area = np.sum(np.sum(f[gps[0]]['ims_id'][:, 0, :, :], axis = 2), axis = 1)
            ims_id = f[gps[0]]['ims_id'][:, 0,]
        ims_id_area = np.sum(np.sum(ims_id, axis = 2), axis = 1)
        plt.plot(ims_id_area[:N])
    plt.title("Grain Area Over Time (%s)" % ic_shape)
    plt.xlabel('Time Frame')
    plt.ylabel('Grain Area')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/grain area over time(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
def make_merge_videos(hps, sub_folder, figname, ic_shape, legend, gps='sim0'):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' a list if it isn't already
    if type(hps)!=list: hps = [hps]
    if type(gps)!=list: gps = [gps]
    
    # Make sure all needed datasets exist
    #dts=['ims_id', 'ims_miso', 'ims_miso_spparks']
    #check_exist_h5(hps, gps, dts)  
    ims_list = []
    for i in tqdm(range(len(hps)), "Making videos"):
        with h5py.File(hps[i], 'a') as f:
            g = f[gps[0]]
            ims = g['ims_id'][:,0]
            ims_list.append(ims)
    ims_merge = sum(ims_list)/len(ims_list)
    ims_merge = (255/np.max(ims_merge)*ims_merge).astype(np.uint8)
    imageio.mimsave('./plots/%s/%s_ims.mp4'%(sub_folder, figname), ims_merge)
    imageio.mimsave('./plots/%s/%s_ims.gif'%(sub_folder, figname), ims_merge)
    
  
    ims_id_area = np.sum(np.sum(ims_merge, axis = 2), axis = 1)
    N = 1200
    plt.figure()  
    plt.plot(ims_id_area[:N])
    plt.title("Grain Area Over Time (%s)" % ic_shape)
    plt.xlabel('Time Frame')
    plt.ylabel('Grain Area')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/grain area over time(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
def make_time_plots(hps, sub_folder, figname, ic_shape, legend = [], gps='last', scale_ngrains_ratio=0.05, cr=None):
    # Run "compute_grain_stats" before this function
    
    #Make 'hps' and 'gps' a list if it isn't already, and set default 'gps'
    if type(hps)!=list: hps = [hps]
    
    if gps=='last':
        gps = []
        for hp in hps:
            with h5py.File(hp, 'r') as f:
                gps.append(list(f.keys())[-1])
                print(f.keys())
        print('Last groups in each h5 file chosen:')
        #print(gps)        
    else:
        if type(gps)!=list: gps = [gps]
    
    # Establish color table
    c = [mcolors.TABLEAU_COLORS[n] for n in list(mcolors.TABLEAU_COLORS)]
    if np.all(cr!=None): #repeat the the color labels using "cr"
        tmp = []
        for i, e in enumerate(c[:len(cr)]): tmp += cr[i]*[e]
        c = tmp
    
    # Make sure all needed datasets exist
    #dts=['grain_areas', 'grain_sides', 'ims_miso', 'ims_miso_spparks']
    #check_exist_h5(hps, gps, dts)  
    
    # Calculate scale limit
    with h5py.File(hps[0], 'r') as f:
        g = f[gps[0]]
        #print(g.keys())
        total_area = np.product(g['ims_id'].shape[1:])
        ngrains = g['grain_areas'].shape[1]
        lim = total_area/(ngrains*scale_ngrains_ratio)
    
    # Plot average grain area through time and find linear slopes
    log = []
    ys = []
    ps = []
    rs = []
    for i in tqdm(range(len(hps)),'Calculating avg grain areas'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas_avg = f[gps[i]+'/grain_areas_avg'][:]
        log.append(grain_areas_avg)
        
        x = np.arange(len(grain_areas_avg))
        p = np.polyfit(x, grain_areas_avg, 1)
        ps.append(p)
        
        fit_line = np.sum(np.array([p[j]*x*(len(p)-j-1) for j in range(len(p))]), axis=0)
        ys.append(fit_line)
        
        r = np.corrcoef(grain_areas_avg, fit_line)[0,1]**2
        rs.append(r)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area (%s)'%ic_shape)
    plt.xlabel('Number of frames')
    plt.ylabel('Average area (pixels)')
    if legend!= []: plt.legend(legend)
    plt.savefig('./plots/%s/avg_grain_area_time(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()

    # Plot scaled average grain area through time and find linear slopes
    ys = []
    ps = []
    rs = []
    si = []
    xs = []
    for i in range(len(hps)):
        grain_areas_avg = log[i]
        ii = np.argmin(np.abs(grain_areas_avg-lim))
        si.append(ii)

        x = np.arange(len(grain_areas_avg))
        p = np.polyfit(x, grain_areas_avg, 1)
        ps.append(p)
        
        fit_line = np.sum(np.array([p[j]*x*(len(p)-j-1) for j in range(len(p))]), axis=0)
        ys.append(fit_line)
        
        r = np.corrcoef(grain_areas_avg, fit_line)[0,1]**2
        rs.append(r)
        
        xx = np.linspace(ngrains,int(ngrains*scale_ngrains_ratio),ii)
        xs.append(xx)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('Slope: %.3f | R2: %.3f'%(ps[i][0], rs[i]))
    plt.title('Average grain area (scaled) (%s)'%ic_shape)
    plt.xlabel('Number of grains')
    plt.ylabel('Average area (pixels)')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/avg_grain_area_time_scaled(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
    # Plot average grain sides through time
    log = []
    for i in tqdm(range(len(hps)),'Plotting avg grain sides'):
        with h5py.File(hps[i], 'r') as f: 
            grain_sides_avg = f[gps[i]+'/grain_sides_avg'][:]
        log.append(grain_sides_avg)
    
    plt.figure()
    for i in range(len(hps)):
        plt.plot(log[i], c=c[i%len(c)])
        legend.append('')
    plt.title('Average number of grain sides (%s)'%ic_shape)
    plt.xlabel('Number of frames')
    plt.ylabel('Average number of sides')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/avg_grain_sides_time(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
    # Plot scaled average grain sides through time
    plt.figure()
    for i in range(len(hps)):
        plt.plot(xs[i], log[i][:len(xs[i])], c=c[i%len(c)])
        plt.xlim([np.max(xs[i]), np.min(xs[i])])
        legend.append('')
    plt.title('Average number of grain sides (scaled) (%s)'%ic_shape)
    plt.xlabel('Number of grains')
    plt.ylabel('Average number of sides')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/avg_grain_sides_time_scaled(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
    # Plot grain size distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating normalized radius distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        ii = (ng<tg).argmax()
        ga = grain_areas[ii]
        gr = np.sqrt(ga/np.pi)
        bins=np.linspace(0,3,10)
        gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean(), bins)
        plt.plot(bins[:-1], gr_dist/gr_dist.sum()/bins[1])
    plt.title('Normalized radius distribution (%d%% grains remaining %s)'%(100*frac, ic_shape))
    plt.xlabel('R/<R>')
    plt.ylabel('Frequency')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/normalized_radius_distribution(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()
    
    # Plot number of sides distribution
    plt.figure()
    frac = 0.25
    for i in tqdm(range(len(hps)),'Calculating number of sides distribution'):
        with h5py.File(hps[i], 'r') as f: 
            grain_areas = f[gps[i]+'/grain_areas'][:]
            grain_sides = f[gps[i]+'/grain_sides'][:]
        tg = (grain_areas.shape[1])*frac
        ng = (grain_areas!=0).sum(1)
        ii = (ng<tg).argmax()
        gs = grain_sides[ii]
        bins=np.arange(3,9)+0.5
        gs_dist, _ = np.histogram(gs[gs!=0], bins)
        plt.plot(bins[1:]-0.5, gs_dist/gs_dist.sum())
    plt.title('Number of sides distribution (%d%% grains remaining %s)'%(100*frac, ic_shape))
    plt.xlabel('Number of sides')
    plt.ylabel('Frequency')
    if legend!=[]: plt.legend(legend)
    plt.savefig('./plots/%s/number_sides_distribution(%s)'%(sub_folder, figname), dpi=300)
    plt.show()
    plt.close()   
    
    
'''-------------------------------------------------------------------------'''

data_type_dict = {"spparks": "SPPARKS", "PRIMME": "PRIMME", "Analytical Solution Eq.(3)": "Analytical Solution Eq.(3)",
                  'step(even_mixed_0)': 'M-PRIMME', 'step(even_mixed_1)': 'M-PRIMME(Mix--1)', 'step(even_mixed_2)': 'M-PRIMME(Mix--2)',
                  'step(even_mixed_4)': 'M-PRIMME(Mix--4)', 'step(even_mixed_6)': 'M-PRIMME(Mix--6)', 'step(even_mixed_8)': 'M-PRIMME(Mix--8)', 
                  'step(even_mixed_10)': 'M-PRIMME(Mix--10)', 'step(even_mixed_12)': 'M-PRIMME(Mix--12)', 'step(even_mixed_14)': 'M-PRIMME(Mix--14)',
                  'step(even_mixed_16)': 'M-PRIMME(Mix--16)'}
ic_shape_dict = {"circular(257_257_64)": "Circular Grain (257-257-128)", "square(512_512_64)": "Square Grain (512-512-128)", 
                 "grain(512_512_512)": "Multi-Grain (512-512-512)", "circular(512_512_200)": "Circular Grain (512-512-400)", 
                 "square(512_512_200)": "Square Grain (512-512-400)",  "grain(1024_1024_4096)": "Multi-Grain (1024-1024-4096)"}

def comp_simple_grain_growth_method(data_dict, sub_folder):

    ic_shape =  "circular(257_257_64)" #"square(512_512_64)" # "circular(257_257_64)"
    paras_dict = {'figname': "Grain Growth (%s) Methods (Step)" % ic_shape, 'fontsize': 18,
                  'figsize': (9, 9), 'method': ['step(4_1)', 'step(4_2)', 'step(4_4)', 'step(4_6)', 'step(4_8)', 'step(4_10)',
                                                'step(4_12)', 'step(4_14)', 'step(4_16)']} # method': ["spparks", "PRIMME", 'step(even_mixed_0)']
   
    # 'method': ['step(even_mixed_1)', 'step(even_mixed_2)', 'step(even_mixed_4)',    'step(even_mixed_6)', 'step(even_mixed_8)', 'step(even_mixed_10)',  'step(even_mixed_12)', 'step(even_mixed_14)', 'step(even_mixed_16)'    
    
    N = 100
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(paras_dict['method']):
        plt.subplot(3, 3, i+1)
        plt.imshow(data_dict[ic_shape][key_data]['ims_id'][N])
        #plt.title("(step %s)"%(N+1), fontsize = paras_dict['fontsize'])
        #plt.title(data_type_dict[key_data], fontsize = paras_dict['fontsize'])
        plt.title(r"M-PRIMME ($N_{\tau}$ = %s)" % key_data.split("_")[-1][:-1], fontsize = paras_dict['fontsize'])
        plt.axis('off')
        #plt.xticks([], [])
        #plt.yticks([], [])   
        #plt.ylabel(data_type_dict[key_data], fontsize = paras_dict['fontsize'])
    plt.subplots_adjust(left = 0.02, right = 0.98, top = 0.97, bottom = 0.01, wspace = 0.1, hspace = 0.15)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.show()
    plt.close()
    
    '''---------------------------------------------------------------------'''
 
    ic_shape =  "circular(257_257_64)"  # "circular(257_257_64)", "square(512_512_64)" 
    paras_dict = {'figname': "Grain Growth (%s) Methods (Time)" % ic_shape, 'fontsize': 18,
                  'figsize': (9, 12), 'method': ["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']}    
    
    N = 275
    time_align_theory = {}
    for i, key_data in enumerate(paras_dict['method']):
        S_cir = np.sum(data_dict["circular(257_257_64)"][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        X = np.arange(0, N).reshape(-1, 1);
        y = np.sum(np.sum(data_dict["circular(257_257_64)"][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        reg = LinearRegression().fit(X, y)    
        time_align_theory[key_data] = t_end / (-reg.intercept_/reg.coef_)    
    
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(paras_dict['method']):
        for j, time_slice in enumerate([0, 30, 60]):
            if time_slice == 0:
                N = 0
            else:
                N = int(np.ceil(time_slice/time_align_theory[key_data]))
            plt.subplot(4, 3, 3*i+j+1)
            plt.imshow(data_dict[ic_shape][key_data]['ims_id'][N])
            plt.title(data_type_dict[key_data] + "--%.1f s"%(N*time_align_theory[key_data][0]), fontsize = paras_dict['fontsize'])
            plt.axis('off')
    plt.subplots_adjust(left = 0.02, right = 0.98, top = 0.97, bottom = 0.01, wspace = 0.1, hspace = 0.15)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.show()
    plt.close()
    
def comp_grain_growth_method(data_dict, sub_folder):
    
    ic_shape = "grain(512_512_512)" #  "grain(512_512_512)", "grain(1024_1024_4096)"
    paras_dict = {'figname': "Grain Growth (%s) Methods (Step)" % ic_shape, 'fontsize': 18,
                  'figsize': (18, 18), 'method': ["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)'],
                  'num_grain': {}}
    
    for key_data in paras_dict['method']:
        paras_dict['num_grain'][key_data] = []
        N = data_dict[ic_shape][key_data]['ims_id'].shape[0]
        for i in range(N):
            grain_ID = np.unique(data_dict[ic_shape][key_data]['ims_id'][i])
            paras_dict['num_grain'][key_data].append(len(grain_ID))
        paras_dict['num_grain'][key_data] = np.array(paras_dict['num_grain'][key_data])
    
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(paras_dict['method']):
        for j, num_grain in enumerate([512, 300, 150, 50]):
            N = (np.abs(paras_dict['num_grain'][key_data] - num_grain)).argmin()
            plt.subplot(4, 4, 4*i+j+1)
            plt.imshow(data_dict[ic_shape][key_data]['ims_id'][N])
            plt.title(data_type_dict[key_data] + "\n%s grains (step %s)"%(paras_dict['num_grain'][key_data][N], N+1), 
                      fontsize = paras_dict['fontsize'])
            plt.axis('off')
    plt.subplots_adjust(left = 0.02, right = 0.98, top = 0.96, bottom = 0.01, wspace = 0.1, hspace = 0.15)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.show()
    plt.close()  
    
def comp_hex_growth_method(data_dict, sub_folder):
    
    ic_shape = "hex" 
    paras_dict = {'figname': "Grain Growth (%s) Methods (Time)" % ic_shape, 'fontsize': 18,
                  'figsize': (9, 5), 'method': ['step(even_mixed_8)']}
    
    N = 250
    time_align_theory = {}
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        S_cir = np.sum(data_dict["circular(257_257_64)"][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        X = np.arange(0, N).reshape(-1, 1);
        y = np.sum(np.sum(data_dict["circular(257_257_64)"][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        reg = LinearRegression().fit(X, y)    
        time_align_theory[key_data] = t_end / (-reg.intercept_/reg.coef_)    
    
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(paras_dict['method']):
        for j, time_slice in enumerate([0, 300]):
            if time_slice == 0:
                N = 0
            else:
                N = int(np.ceil(time_slice/time_align_theory[key_data]))
            plt.subplot(1, 2, 2*i+j+1)
            plt.imshow(data_dict[ic_shape][key_data]['ims_id'][int(N/10)].T)
            plt.title(data_type_dict[key_data] + "--%.1f s"%(N*time_align_theory[key_data][0]), fontsize = paras_dict['fontsize'])
            plt.axis('off')
    plt.subplots_adjust(left = 0.02, right = 0.98, top = 0.92, bottom = 0.01, wspace = 0.1, hspace = 0.15)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.show()
    plt.close() 

def comp_simple_grain_growth_area_method(data_dict, sub_folder):
    
    ic_shape = "circular(257_257_64)"  # "circular(257_257_64)"  "square(512_512_64)"
    paras_dict = {'figsize': (9, 9), 'figname': "Grain Growth (%s) Area Methods (Step)" % ic_shape, 'fontsize': 18,
                  'ylims': {"circular(257_257_64)": [0, 140], "square(512_512_64)": [0, 165],
                            "circular(512_512_200)": [500, 1300], "square(512_512_200)": [800, 1600]},
                  'yticks': {"circular(257_257_64)": np.arange(0, 141, 20), "square(512_512_64)": np.arange(0, 161, 40),
                             "circular(512_512_200)": np.arange(500, 1301, 200), "square(512_512_200)": np.arange(800, 1601, 200)},
                  'method': ['spparks', 'PRIMME', 'step(4_1)', 'step(4_2)', 'step(4_4)', 'step(4_6)', 'step(4_8)', 'step(4_10)', 'step(4_12)', 'step(4_14)', 'step(4_16)']}

   # 'method': ['spparks', 'PRIMME', 'step(even_mixed_1)', 'step(even_mixed_2)', 'step(even_mixed_4)', 'step(even_mixed_8)', 'step(even_mixed_10)', 'step(even_mixed_12)', 'step(even_mixed_14)', 'step(even_mixed_16)']    

    data_vis_dict = {}
    N = 1800
    for i, key_data in enumerate(paras_dict['method']):
        if "grain" not in ic_shape:
            data_vis_dict[key_data] = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        else:
            data_vis_dict[key_data] = data_dict[ic_shape][key_data]['grain_areas_avg'][:N]/100
        
    x = np.arange(N)
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(data_vis_dict.keys()):
        if key_data in ['spparks','PRIMME']:
            plt.plot(x, data_vis_dict[key_data], label = data_type_dict[key_data], c=c[i%len(c)], linewidth=3) # 
        else:
            plt.plot(x, data_vis_dict[key_data], label = r"M-PRIMME ($N_{\tau}$ = %s)" % key_data.split("_")[-1][:-1], c=c[i%len(c)], linewidth=3)
    plt.xlabel('Inference Step', fontsize = paras_dict['fontsize'])
    plt.ylabel(r"$\langle R \rangle ^2(\times 10^{-10} m^2)$", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim(paras_dict['ylims'][ic_shape])
    plt.yticks(paras_dict['yticks'][ic_shape], paras_dict['yticks'][ic_shape], fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'], bbox_to_anchor=(0.5, -0.1),
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)         
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.3, wspace = 0.25, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()          
    
    '''---------------------------------------------------------------------'''
    
    ic_shape =  "square(512_512_64)" # "circular(257_257_64)"  "square(512_512_64)"
    paras_dict = {'figsize': (9, 9), 'figname': "Grain Growth (%s) Area Methods" % ic_shape, 'fontsize': 18,
                  'ylims': {"circular(257_257_64)": [0, 140], "square(512_512_64)": [0, 165],
                            "circular(512_512_200)": [500, 1300], "square(512_512_200)": [800, 1600]},
                  'yticks': {"circular(257_257_64)": np.arange(0, 141, 20), "square(512_512_64)": np.arange(0, 161, 40),
                             "circular(512_512_200)": np.arange(500, 1301, 200), "square(512_512_200)": np.arange(800, 1601, 200)}}
    time_align_theory = {}
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        S_cir = np.sum(data_dict[ic_shape][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        areas = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'], axis = 2)/100, axis = 1)
        N = np.where(areas > 100)[0][-1]
        X = np.arange(0, N).reshape(-1, 1);
        y = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        reg = LinearRegression().fit(X, y)    
        time_align_theory[key_data] = t_end / (-reg.intercept_/reg.coef_)        
    
    data_vis_dict = {}
    if "circular" in ic_shape:
        S_cir = np.sum(data_dict[ic_shape][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        x = np.arange(0, t_end, 0.1)
        data_vis_dict["Analytical Solution Eq.(3)"] = [x, S_cir - 2*np.pi*3.24*0.74*0.1*x]
  
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        areas = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'], axis = 2)/100, axis = 1)
        N = np.where(areas > 1)[0][-1]
        data_vis_dict[key_data] = [np.arange(N), areas[:N]]
        
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(data_vis_dict.keys()):
        if key_data == "Analytical Solution Eq.(3)":
            plt.plot(data_vis_dict[key_data][0], data_vis_dict[key_data][1], label = data_type_dict[key_data], c=c[i%len(c)])
        else:   
            plt.plot(data_vis_dict[key_data][0] * time_align_theory[key_data], data_vis_dict[key_data][1], label = data_type_dict[key_data], c=c[i%len(c)])
    plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
    plt.ylabel(r"$\langle R \rangle ^2(\times 10^{-10} m^2)$", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim(paras_dict['ylims'][ic_shape])
    plt.yticks(paras_dict['yticks'][ic_shape], paras_dict['yticks'][ic_shape], fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'], bbox_to_anchor=(0.5, -0.1),
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)         
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.25, wspace = 0.25, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()      
         
    '''---------------------------------------------------------------------'''    
    ic_shape = "square(512_512_200)" #"circular(257_257_64)" "circular(512_512_200)" "square(512_512_64)", "square(512_512_200)"
    paras_dict['figname'] = "Grain Growth (%s) Area Methods" % ic_shape
    N = 600
    time_align_theory = {}
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        S_cir = np.sum(data_dict[ic_shape][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        X = np.arange(0, N).reshape(-1, 1);
        y = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        reg = LinearRegression().fit(X, y)    
        time_align_theory[key_data] = t_end / (-reg.intercept_/reg.coef_)        
    
    data_vis_dict = {}
    if "circular" in ic_shape:
        S_cir = np.sum(data_dict[ic_shape][key_data]['ims_id'][0])/100
        t_end = (S_cir-500) / (2*np.pi*3.24*0.74*0.1)
        x = np.arange(0, t_end, 0.1)
        data_vis_dict["Analytical Solution Eq.(3)"] = [x, S_cir - 2*np.pi*3.24*0.74*0.1*x]
        
    N = 1800
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        if "grain" not in ic_shape:
            data_vis_dict[key_data] = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        else:
            data_vis_dict[key_data] = data_dict[ic_shape][key_data]['grain_areas_avg'][:N]/100
        
    x = np.arange(N)
    plt.figure(figsize = paras_dict['figsize'])
    for i, key_data in enumerate(data_vis_dict.keys()):
        if key_data == "Analytical Solution Eq.(3)":
            plt.plot(data_vis_dict[key_data][0], data_vis_dict[key_data][1], label = data_type_dict[key_data], c=c[i%len(c)])
        else:   
            plt.plot(x * time_align_theory[key_data], data_vis_dict[key_data], label = data_type_dict[key_data], c=c[i%len(c)])
    plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
    plt.ylabel(r"$\langle R \rangle ^2(\times 10^{-10} m^2)$", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim(paras_dict['ylims'][ic_shape])
    plt.yticks(paras_dict['yticks'][ic_shape], paras_dict['yticks'][ic_shape], fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'], bbox_to_anchor=(0.5, -0.1),
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)         
    plt.subplots_adjust(left = 0.13, right = 0.95, top = 0.95, bottom = 0.25, wspace = 0.25, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()   

def comp_grain_growth_area_method(data_dict, sub_folder):
    
    ic_shape = "circular(257_257_64)" #"circular(257_257_64)" "circular(512_512_200)" "square(512_512_64)", "square(512_512_200)"
    N = 300
    time_est = {}
    for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
        S_cir = np.sum(data_dict[ic_shape][key_data]['ims_id'][0])/100
        t_end = S_cir / (2*np.pi*3.24*0.74*0.1)
        X = np.arange(0, N).reshape(-1, 1);
        y = np.sum(np.sum(data_dict[ic_shape][key_data]['ims_id'][:N], axis = 2)/100, axis = 1)
        reg = LinearRegression().fit(X, y)    
        time_est[key_data] = t_end / (-reg.intercept_/reg.coef_) 

    ic_shape = "grain(1024_1024_4096)" #  "grain(512_512_512)", "grain(1024_1024_4096)"
    paras_dict = {'figsize': (18, 6), 'figname': "Grain Growth (%s) Area and Grain Number Methods" % ic_shape, 'fontsize': 18,
                  'ylims': {"grain(512_512_512)": [0, 140], "grain(1024_1024_4096)": [0, 8000]},
                  'yticks': {"grain(512_512_512)": np.arange(0, 141, 20), "grain(1024_1024_4096)": np.arange(0, 8001, 2000)},
                  'method': ["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)'],
                  'if_scale': True}
    if paras_dict['if_scale']:
        paras_dict['figname'] =  "Grain Growth (%s) Area and Grain Number Methods (scaled time)" % ic_shape
        

    time_align_theory = {}
    N = 600 
    X = np.arange(0, N).reshape(-1, 1)
    y = data_dict[ic_shape]["spparks"]['grain_areas_avg'][:N]
    reg = LinearRegression().fit(X, y)  
    coef_SPPARKS = reg.coef_.copy()
    for key_data in paras_dict['method']:
        X = np.arange(0, N).reshape(-1, 1)
        y = data_dict[ic_shape][key_data]['grain_areas_avg'][:N]
        reg = LinearRegression().fit(X, y)
        time_align_theory[key_data] = reg.coef_/coef_SPPARKS
        print(key_data, reg.intercept_, reg.coef_)
    
    data_vis_dict = {'grain_areas_avg': {}, 'grain_num': {}, 'grain_sides_avg': {}} 
    N = 600
    for i, key_data in enumerate(paras_dict['method']):
        data_vis_dict['grain_areas_avg'][key_data] = data_dict[ic_shape][key_data]['grain_areas_avg'][:N]
        data_vis_dict['grain_num'][key_data] = []
        for i in range(N):
            grain_ID = np.unique(data_dict[ic_shape][key_data]['ims_id'][i])
            data_vis_dict['grain_num'][key_data].append(len(grain_ID))
        data_vis_dict['grain_num'][key_data] = np.array(data_vis_dict['grain_num'][key_data])   
        data_vis_dict['grain_sides_avg'][key_data] = data_dict[ic_shape][key_data]['grain_sides_avg'][:N]
        
    x = np.arange(N)
    plt.figure(figsize = paras_dict['figsize'])
    plt.subplot(131)
    for i, key_data in enumerate(data_vis_dict['grain_areas_avg'].keys()):
        if paras_dict['if_scale']: 
            plt.plot(x*time_align_theory[key_data]*time_est["spparks"], data_vis_dict['grain_areas_avg'][key_data][:N],
                     label = data_type_dict[key_data], c=c[i%len(c)])
        else:
            plt.plot(x, data_vis_dict['grain_areas_avg'][key_data][:N],
                     label = data_type_dict[key_data], c=c[i%len(c)])
    plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
    plt.ylabel(r"$\langle R \rangle ^2(\times 10^{-12} m^2)$", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim(paras_dict['ylims'][ic_shape])
    plt.yticks(paras_dict['yticks'][ic_shape], paras_dict['yticks'][ic_shape], fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'],
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)   
    plt.subplot(132)
    for i, key_data in enumerate(data_vis_dict['grain_num'].keys()): 
        if paras_dict['if_scale']:
            plt.plot(x*time_align_theory[key_data]*time_est["spparks"], data_vis_dict['grain_num'][key_data][:N],
                    label = data_type_dict[key_data], c=c[i%len(c)])
        else:
            plt.plot(x, data_vis_dict['grain_num'][key_data][:N],
                     label = data_type_dict[key_data], c=c[i%len(c)])
    plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
    plt.ylabel("Number of Grains", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim([0, 4100])
    plt.yticks(np.arange(0, 4100, 1000), np.arange(0, 4100, 1000), fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'],
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)
    plt.subplot(133)
    for i, key_data in enumerate(data_vis_dict['grain_sides_avg'].keys()):
        if paras_dict['if_scale']: 
            plt.plot(x*time_align_theory[key_data]*time_est["spparks"], data_vis_dict['grain_sides_avg'][key_data][:N],
                     label = data_type_dict[key_data], c=c[i%len(c)])
        else:
            plt.plot(x, data_vis_dict['grain_sides_avg'][key_data][:N],
                     label = data_type_dict[key_data], c=c[i%len(c)])
    plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
    plt.ylabel(r"$Mean number of sides \langle R \rangle$", fontsize = paras_dict['fontsize'])
    plt.xticks(fontsize = paras_dict['fontsize'])
    plt.ylim([5.9, 6.1])
    plt.yticks(fontsize = 16)
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'],
               fancybox = False, shadow = False, ncol = 2, frameon = False, markerscale = 10)      
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.15, wspace = 0.15, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()   
   
    '''---------------------------------------------------------------------'''
    ic_shape = "grain(1024_1024_4096)" #  "grain(512_512_512)", "grain(1024_1024_4096)"
    paras_dict = {'figsize': (18, 6), 'figname': 'Normalized radius distribution (%s)'%(ic_shape), 
                  'fontsize': 18, 'num_grain': [2000, 1000, 500],
                  'method': ["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)'],
                  'if_scale': False}
    if paras_dict['if_scale']:
        paras_dict['figname'] =  'Normalized radius distribution (%s)'%(ic_shape)
      
    N = 1200
    data_vis_dict = {}
    for num_grain in paras_dict['num_grain']:
        data_vis_dict[num_grain] = {}
        for i, key_data in enumerate(paras_dict['method']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas']
            ng = (grain_areas!=0).sum(1)
            ii = (ng<num_grain).argmax()
            ga = grain_areas[ii]
            gr = np.sqrt(ga/np.pi)
            bins = np.linspace(0,3,20)
            gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean(), bins)        
            data_vis_dict[num_grain][key_data] = [bins, gr_dist]
   
    plt.figure(figsize = paras_dict['figsize'])
    for i, num_grain in enumerate(data_vis_dict.keys()): 
        plt.subplot(1, 3, i+1)
        for j,key_data in enumerate(data_vis_dict[num_grain].keys()):
            plt.plot(data_vis_dict[num_grain][key_data][0][:-1], 
                     data_vis_dict[num_grain][key_data][1]/data_vis_dict[num_grain][key_data][1].sum()/data_vis_dict[num_grain][key_data][0][1], 
                     label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel('R/<R> - Normalized radius', fontsize = paras_dict['fontsize'])
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        plt.yticks(fontsize = paras_dict['fontsize'])
        plt.legend(loc = 'upper right', scatterpoints = 1, fontsize = paras_dict['fontsize'],
                   fancybox = False, shadow = False, ncol = 1, frameon = False, markerscale = 10)          
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.15, wspace = 0.25, hspace = 0.2)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()         
       
    '''---------------------------------------------------------------------'''

    ic_shape = "grain(1024_1024_4096)" #  "grain(512_512_512)", "grain(1024_1024_4096)"
    paras_dict = {'figsize': (18, 6), 'figname': 'Normalized sides distribution (%s)'%(ic_shape), 
                  'fontsize': 18, 'num_grain': [2000, 1000, 500],
                  'method': ["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)'],
                  'if_scale': False}
    if paras_dict['if_scale']:
        paras_dict['figname'] = 'Normalized sides distribution (%s)'%(ic_shape)
    
    data_vis_dict = {}
    for num_grain in paras_dict['num_grain']:
        data_vis_dict[num_grain] = {}
        for i, key_data in enumerate(["spparks", "PRIMME", 'step(even_mixed_4)', 'step(even_mixed_8)']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas'][:N]
            grain_sides = data_dict[ic_shape][key_data]['grain_sides'][:N]
            ng = (grain_areas!=0).sum(1)
            ii = (ng<num_grain).argmax()
            gs = grain_sides[ii]
            bins = np.arange(3, 9)+0.5
            gs_dist, _ = np.histogram(gs[gs!=0], bins)        
            data_vis_dict[num_grain][key_data] = [bins, gs_dist]

    plt.figure(figsize = (18, 9))
    for i, num_grain in enumerate(data_vis_dict.keys()): 
        plt.subplot(1, 3, i+1)
        for j,key_data in enumerate(data_vis_dict[num_grain].keys()):
            plt.plot(data_vis_dict[num_grain][key_data][0][1:]-0.5, data_vis_dict[num_grain][key_data][1]/data_vis_dict[num_grain][key_data][1].sum(), 
                     label = data_type_dict[key_data], c=c[j%len(c)])
        plt.title(ic_shape_dict[ic_shape], fontsize = paras_dict['fontsize']) 
        plt.xlabel('Number of sides', fontsize = paras_dict['fontsize'])
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        plt.yticks(fontsize = paras_dict['fontsize'])        
        plt.legend(loc = 'upper right', scatterpoints = 1, fontsize = paras_dict['fontsize'],
                   fancybox = False, shadow = False, ncol = 1, frameon = False, markerscale = 10)          
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.15, wspace = 0.25, hspace = 0.2)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()    

'''-------------------------------------------------------------------------'''

def grain_growth_statistics(data_dict, sub_folder):

    data_vis_dict = {}
    N = 400; frac = 0.25
    for ic_shape in ["grain(512_512_512)", "grain(1024_1024_4096)"]:    
        data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", "PRIMME", 'step(1_1)', 'step(2_2)', 'step(3_3)', 'step(4_4)', 'step(5_5)', 'step(6_6)']):
            grain_sides_avg = data_dict[ic_shape][key_data]['grain_sides_avg'][:N]     
            data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]][key_data] = grain_sides_avg
    
        data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", 'step(1_1)', 'step(2_2)', 'step(3_3)', 'step(4_4)', 'step(5_5)', 'step(6_6)', 'step(7_7)']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas'][:N]
            tg = (grain_areas.shape[1])*frac
            ng = (grain_areas!=0).sum(1)
            ii = (ng<tg).argmax()
            ga = grain_areas[ii]
            gr = np.sqrt(ga/np.pi)
            bins = np.linspace(0,3,10)
            gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean()+0.25, bins)        
            data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data] = [bins, gr_dist]
    
        data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", "PRIMME", 'step(1_1)', 'step(2_2)', 'step(3_3)', 'step(4_4)', 'step(5_5)', 'step(6_6)']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas'][:N]
            grain_sides = data_dict[ic_shape][key_data]['grain_sides'][:N]
            tg = (grain_areas.shape[1])*frac
            ng = (grain_areas!=0).sum(1)
            ii = (ng<tg).argmax()
            gs = grain_sides[ii]
            bins = np.arange(3, 12) + 0.5
            gs_dist, _ = np.histogram(gs[gs!=0]+1, bins)       
            data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data] = [bins, gs_dist]
    
    
    paras_dict = {'figname': 'Statistics_Time Step_1_7(%s)'% N, 'fontsize': 19, 
                  'yticks': {"Average Grain Sides": np.array([5.8, 5.9, 6.0, 6.1, 6.2]), "Normalized Radius": np.array([0, 0.3, 0.6, 0.9, 1.2]),
                             "Number of Sides": np.array([0, 0.1, 0.2, 0.3, 0.4])},
                  'ylims': {"Average Grain Sides": [5.8, 6.2], "Normalized Radius": [0, 1.2],
                             "Number of Sides": [0, 0.4]}}
    
    plt.figure(figsize = (18, 9))
    for i, ic_shape in enumerate(["grain(512_512_512)", "grain(1024_1024_4096)"]): 
        plt.subplot(2, 3, 3*i+1)
        for j, key_data in enumerate(data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]].keys()):
            plt.plot(data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]][key_data], label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
        plt.ylabel("Mean number of sides" + r"$\langle F \rangle$", fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Average Grain Sides"])
        plt.yticks(paras_dict['yticks']["Average Grain Sides"], paras_dict['yticks']["Average Grain Sides"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        
        plt.subplot(2, 3, 3*i+2)
        for j, key_data in enumerate(data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]].keys()):
            x = data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][0]
            y = data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][1]
            plt.plot(x[:-1], y/y.sum()/x[1], label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel("R/" + r"$\langle R \rangle$" + "Normalized Radius", fontsize = paras_dict['fontsize'])        
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Normalized Radius"])
        plt.yticks(paras_dict['yticks']["Normalized Radius"], paras_dict['yticks']["Normalized Radius"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        
        plt.subplot(2, 3, 3*i+3)
        for j, key_data in enumerate(data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]].keys()):
            x = data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][0]
            y = data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][1]
            plt.plot(x[:-1]-0.5, y/y.sum(), label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel( "Number of Sides", fontsize = paras_dict['fontsize'])  
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Number of Sides"])
        plt.yticks(paras_dict['yticks']["Number of Sides"], paras_dict['yticks']["Number of Sides"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'], bbox_to_anchor=(-0.8, -0.25),
               fancybox = False, shadow = False, ncol = 4, frameon = False, markerscale = 10)         
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.25, wspace = 0.25, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()   
    
    
    data_vis_dict = {}
    N = 400; frac = 0.25
    for ic_shape in ["grain(512_512_512)", "grain(1024_1024_4096)"]:    
        data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", 'step(8_2)', 'step(8_4)', 'step(8_6)', 'step(8_8)', 'step(8_10)', 'step(8_12)', 'step(8_14)']):
            grain_sides_avg = data_dict[ic_shape][key_data]['grain_sides_avg'][:N]     
            data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]][key_data] = grain_sides_avg
    
        data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", 'step(8_2)', 'step(8_4)', 'step(8_6)', 'step(8_8)', 'step(8_10)', 'step(8_12)', 'step(8_14)']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas'][:N]
            tg = (grain_areas.shape[1])*frac
            ng = (grain_areas!=0).sum(1)
            ii = (ng<tg).argmax()
            ga = grain_areas[ii]
            gr = np.sqrt(ga/np.pi)
            bins = np.linspace(0,3,10)
            gr_dist, _ = np.histogram(gr[gr!=0]/gr[gr!=0].mean()+0.2, bins)        
            data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data] = [bins, gr_dist]
    
        data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]] = {}
        for i, key_data in enumerate(["spparks", 'step(8_2)', 'step(8_4)', 'step(8_6)', 'step(8_8)', 'step(8_10)', 'step(8_12)', 'step(8_14)']):
            grain_areas = data_dict[ic_shape][key_data]['grain_areas'][:N]
            grain_sides = data_dict[ic_shape][key_data]['grain_sides'][:N]
            tg = (grain_areas.shape[1])*frac
            ng = (grain_areas!=0).sum(1)
            ii = (ng<tg).argmax()
            gs = grain_sides[ii]
            bins = np.arange(3, 12) + 0.5
            gs_dist, _ = np.histogram(gs[gs!=0]+1, bins)       
            data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data] = [bins, gs_dist]
    
    
    paras_dict = {'figname': 'Statistics_Time Step_8_2_14(%s)'% N, 'fontsize': 19, 
                  'yticks': {"Average Grain Sides": np.array([5.8, 5.9, 6.0, 6.1, 6.2]), "Normalized Radius": np.array([0, 0.3, 0.6, 0.9, 1.2]),
                             "Number of Sides": np.array([0, 0.1, 0.2, 0.3, 0.4])},
                  'ylims': {"Average Grain Sides": [5.8, 6.2], "Normalized Radius": [0, 1.2],
                             "Number of Sides": [0, 0.4]}}
    
    
    plt.figure(figsize = (18, 9))
    for i, ic_shape in enumerate(["grain(512_512_512)", "grain(1024_1024_4096)"]): 
        plt.subplot(2, 3, 3*i+1)
        for j, key_data in enumerate(data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]].keys()):
            plt.plot(data_vis_dict["  Average Grain Sides\n" + ic_shape_dict[ic_shape]][key_data], label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel('Time (sec)', fontsize = paras_dict['fontsize'])
        plt.ylabel("Mean number of sides" + r"$\langle F \rangle$", fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Average Grain Sides"])
        plt.yticks(paras_dict['yticks']["Average Grain Sides"], paras_dict['yticks']["Average Grain Sides"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        
        plt.subplot(2, 3, 3*i+2)
        for j, key_data in enumerate(data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]].keys()):
            x = data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][0]
            y = data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][1]
            plt.plot(x[:-1], y/y.sum()/x[1], label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel("R/" + r"$\langle R \rangle$" + "Normalized Radius", fontsize = paras_dict['fontsize'])        
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Normalized Radius"])
        plt.yticks(paras_dict['yticks']["Normalized Radius"], paras_dict['yticks']["Normalized Radius"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
        
        plt.subplot(2, 3, 3*i+3)
        for j, key_data in enumerate(data_vis_dict["Normalized Radius Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]].keys()):
            x = data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][0]
            y = data_vis_dict["Normalized Sides Distribution (%d%% Grains)\n"%(100*frac) + ic_shape_dict[ic_shape]][key_data][1]
            plt.plot(x[:-1]-0.5, y/y.sum(), label = data_type_dict[key_data], c=c[j%len(c)])
        plt.xlabel( "Number of Sides", fontsize = paras_dict['fontsize'])  
        plt.ylabel('Frequency', fontsize = paras_dict['fontsize'])
        plt.ylim(paras_dict['ylims']["Number of Sides"])
        plt.yticks(paras_dict['yticks']["Number of Sides"], paras_dict['yticks']["Number of Sides"], fontsize = paras_dict['fontsize'])
        plt.xticks(fontsize = paras_dict['fontsize'])
    
    plt.legend(loc = 'upper center', scatterpoints = 1, fontsize = paras_dict['fontsize'], bbox_to_anchor=(-0.8, -0.25),
               fancybox = False, shadow = False, ncol = 4, frameon = False, markerscale = 10)         
    plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, bottom = 0.25, wspace = 0.25, hspace = 0.3)
    plt.savefig(Path('./plots/%s/%s'%(sub_folder, paras_dict['figname'])).with_suffix('.png'), dpi=300)
    plt.close()   



















