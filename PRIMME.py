#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION:
    This script defines the PRIMME simulator class used to simulate microstructural grain growth
    The neural network model used to predict the action likelihood is written in Tensorflow (Keras)
    The functions besides of the model are written in Pytorch to parallelize the operations using GPUs if available
    This class must be passed a SPPARKS class ('env'), which provides an initial condition, and training data, features, and labels 
    The main functions of the class include predicting the action likelihood (given an intial condition) and training the model (given features and labels)

IF THIS CODE IS USED FOR A RESEARCH PUBLICATION, please cite (https://arxiv.org/abs/2203.03735): 
    Yan, Weishi, et al. "Predicting 2D Normal Grain Growth using a Physics-Regularized Interpretable Machine Learning Model." arXiv preprint arXiv:2203.03735 (2022).
"""

# IMPORT LIBRARIES
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import torch.utils.data as Data
from random import shuffle
import gc
import functions as fs
from tqdm import tqdm
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device  = 'cpu'
# BUILD PRIMME CLASS

class PRIMME(nn.Module):
    def __init__(self, obs_dim=17, act_dim=17, pad_mode="circular", pad_value=-1, learning_rate=0.00005, reg=1, num_dims=2, 
                 mode = "Single_Step", epoch = 10, device = "cpu"):
        super(PRIMME, self).__init__()

        # ESTABLISH PARAMETERS USED BY CLASS
        #self.device = torch.device("cpu")       # Device for processing
        self.device = device       # Device for processing
        self.obs_dim = obs_dim                  # Observation (input) size (one side of a square)
        self.act_dim = act_dim                  # Action (output) size (one side of a square)
        self.pad_mode = pad_mode                # Padding mode ("circular" or "reflect")
        self.pad_value = pad_value
        self.learning_rate = learning_rate      # Learning rate
        self.reg = reg
        self.num_dims = num_dims                # Number of Dimensions (2 or 3)
        self.mode = mode        
        self.epoch = epoch

        # ESTABLISH VARIABLES TRACKED
        self.training_loss = []                 # Training loss (history)
        self.training_acc = []                  # Training Accuracy (history)
        self.validation_loss = []
        self.validation_acc = []
        self.seq_samples = []
        self.im_seq_T = []

        # DEFINE NEURAL NETWORK
        self.f1 = nn.Linear(self.obs_dim ** self.num_dims, 21 * 21 * 4)
        self.f2 = nn.Linear(21 * 21 * 4, 21 * 21 * 2)
        self.f3 = nn.Linear(21 * 21 * 2, 21 * 21)
        self.f4 = nn.Linear(21 * 21, self.act_dim ** self.num_dims)
        self.dropout = nn.Dropout(p = 0.25) 
        self.BatchNorm1 = nn.BatchNorm1d(21 * 21 * 4)
        self.BatchNorm2 = nn.BatchNorm1d(21 * 21 * 2)
        self.BatchNorm3 = nn.BatchNorm1d(21 * 21)
        self.rnn = nn.RNN(input_size = self.act_dim ** self.num_dims, hidden_size =  self.act_dim ** self.num_dims,
                          num_layers = 2, nonlinearity = 'relu', batch_first = True, dropout = 0.1)

        # DEFINE NEURAL NETWORK OPTIMIZATION
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_func = torch.nn.MSELoss()  # Mean squared error loss
        self.optimizer.zero_grad()  # Make all the gradients zero

    def forward(self, x, h0):
        # def forward: Run input X through the neural network
        #   Inputs--
        #        x: microstructure features around a center pixel
        #   Outputs--
        #       y: "action likelihood" for the center pixel to flip to each the grain associated with each other pixel

        h1 = F.relu(self.f1(x))
        out = self.dropout(h1)   
        out = self.BatchNorm1(out)
        h2 = F.relu(self.f2(out))
        out = self.dropout(h2)
        out = self.BatchNorm2(out)
        h3 = F.relu(self.f3(out))
        out = self.dropout(h3)
        out = self.BatchNorm3(out)
        h4 = self.f4(out)
        #print(h4.shape)
        output, h0 = self.rnn(h4.reshape(-1, 1, self.act_dim ** self.num_dims), h0)
        #y  = torch.sigmoid(output)
        y  = torch.relu(output)
        
        return y, h0

    def load_data(self, n_samples, n_step, h5_paths = ['spparks_data_size257x257_ngrain256-256_nsets200_future4_max100_offset1_kt0.h5']):
       
        self.dataset_name = ""
        self.n_step = n_step
        for i, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, 'r') as f:
                print("Keys: %s" % f.keys())  
                im_seq_T = f['ims_id'][:]
            if self.n_step[0] == "even_mixed":
                n_step_list = np.arange(1, 9)
                for j in range(n_samples[i]):
                    if (len(self.im_seq_T) == 0) or (j == 0):
                        self.im_seq_T = im_seq_T[:1, [0, 1]]
                    else:
                        self.im_seq_T  = np.concatenate((self.im_seq_T, im_seq_T[j:j+1, [0, n_step_list[j%8]]]), axis = 0)                
            else:
                if (len(self.im_seq_T) == 0):
                    self.im_seq_T = im_seq_T[:n_samples[i], [0, self.n_step[0]]]
                else:
                    self.im_seq_T  = np.concatenate((self.im_seq_T, im_seq_T[:n_samples[i], [0, self.n_step[0]]]), axis = 0)
                
            if self.dataset_name == "":
                self.dataset_name += h5_path.split("spparks_")[1].split("_kt")[0]
            else:
                self.dataset_name += ("_") + h5_path.split("spparks_")[1].split("_kt")[0]
        self.im_seq_T = torch.from_numpy(self.im_seq_T)
        self.seq_samples = list(np.arange(len(self.im_seq_T)))              
   
    def step(self, im, evaluate=True):
        # def step: Apply one step of growth to microstructure image IM
        #   Inputs--
        #        im: initial microstructure ID image
        #   Outputs--
        #    im_out: new microstructure ID image after one growth step
        features = fs.compute_features(im, obs_dim=self.obs_dim, pad_mode=self.pad_mode, pad_value=self.pad_value)
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]  
        features = features[indx_use,]
        #print(features.shape)
        
        action_features = fs.my_unfoldNd(im, kernel_size=self.act_dim, pad_mode=self.pad_mode, pad_value=self.pad_value)[0,] 
        action_features = action_features[...,indx_use]
        '''
        h0 = torch.zeros(2, len(indx_use), self.obs_dim**2).to(self.device)
        predictions, h0 = self.forward(features.reshape(-1, self.act_dim**self.num_dims), h0)  
        predictions = predictions.reshape(-1, self.act_dim**self.num_dims)
        action_values = torch.argmax(predictions, dim=1)
        '''
        batch_size = 500000
        features_split = torch.split(features, batch_size)
        self.predictions_split = []
        self.action_values_split = []
        for e in features_split:
            h0 = torch.zeros(2, len(e), self.obs_dim**2).to(self.device)
            predictions, h0 = self.forward(e.reshape(-1, self.act_dim**self.num_dims), h0)
            predictions = predictions.reshape(-1, self.act_dim**self.num_dims)
            action_values = torch.argmax(predictions, dim=1)            
            if evaluate==True: 
                self.predictions_split.append(predictions)
            self.action_values_split.append(action_values)
            print(e.shape)
        if evaluate==True: predictions = torch.cat(self.predictions_split, dim=0)
        action_values = torch.hstack(self.action_values_split)
        
        upated_values = torch.gather(action_features, dim=0, index=action_values.unsqueeze(0))[0,]
        im_next_predicted = im.flatten().float()
        
        im_next_predicted[indx_use] = upated_values
        im_next_predicted = im_next_predicted.reshape(im.shape)
        
        return indx_use, predictions, im_next_predicted

    def build_features(self, im_cur):
        
        features = fs.compute_features(im_cur, obs_dim=self.obs_dim, pad_mode=self.pad_mode, pad_value=self.pad_value)
        mid_ix = (np.array(features.shape[1:])/2).astype(int)
        ind = tuple([slice(None)]) + tuple(mid_ix)
        indx_use = torch.nonzero(features[ind])[:,0]        
        features = features[indx_use,]
        #features, self.p = fs.unison_shuffled_copies_single(features)
        h0 = torch.zeros(2, len(indx_use), self.obs_dim**2).to(self.device)
        
        return features, indx_use, h0 

    def build_labels(self, im_seq, indx_use):
        
        labels = fs.compute_labels(im_seq, obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, 
                                   pad_mode=self.pad_mode, pad_value=self.pad_value)  
        labels = labels[indx_use,]
        #labels[p].reshape(-1, self.act_dim**self.num_dims) 
        
        return labels.reshape(-1, self.act_dim**self.num_dims) 

    def train(self, evaluate=True, if_plot=True):
        # def train: Train the PRIMME neural network architecture with self.im_seq. The first image in self.im_seq
        # is used as the initial condition and the last image in self.im_seq is the desired end goal

        #Compute features and labels
        shuffle(self.seq_samples)
        for seq_sample in self.seq_samples:
            self.seq_sample = seq_sample
            self.im_seq = self.im_seq_T[seq_sample].to(self.device)
            self.im_cur = self.im_seq[0:1,]
            self.features, self.indx_use, self.h0 = self.build_features(self.im_cur)  
                
            if evaluate: 
                loss, accuracy = self.compute_metrics()
                self.validation_loss.append(loss.detach().cpu())
                self.validation_acc.append(accuracy.detach().cpu())
                
            loss = torch.zeros(1)[0].to(self.device)
            for i in range(self.n_step[1]):
                outputs, h0 = self.forward(self.features.reshape(-1, self.act_dim**self.num_dims), self.h0)
                _, _, im_next_predicted = self.step(self.im_cur)
                if i == self.n_step[1]-1:
                    im_seq = self.im_seq.detach().clone()
                    im_seq[0] = self.im_cur
                    self.labels = self.build_labels(im_seq.to(self.device), self.indx_use)
                    loss = self.loss_func(outputs.reshape(-1, self.act_dim**self.num_dims), self.labels)      
                self.im_cur = im_next_predicted
                self.features, self.indx_use, self.h0 = self.build_features(self.im_cur)
              
            self.optimizer.zero_grad()  # Zero the gradient
            loss.backward()             # Perform backpropagation
            self.optimizer.step()       # Step with optimizer
    
            if evaluate: 
                loss, accuracy = self.compute_metrics()
                self.training_loss.append(loss.detach().cpu().numpy())
                self.training_acc.append(accuracy.detach().cpu().numpy())
            
            if if_plot:
                self.plot(self.result_path)
        
    def compute_metrics(self):
        
        im_cur = self.im_seq[0:1,]
        for i in range(self.n_step[1]):
            indx_use, predictions, im_next_predicted = self.step(im_cur)
            if i == self.n_step[1]-1:
                im_seq = self.im_seq.detach().clone()
                im_seq[0] = self.im_cur ### pay attention to it
            im_cur = im_next_predicted
        im_next = self.im_seq[-2:-1, ]
        labels = fs.compute_labels(im_seq.to(self.device), obs_dim=self.obs_dim, act_dim=self.act_dim, reg=self.reg, pad_mode=self.pad_mode)  
        labels = labels[indx_use,]        
        accuracy = torch.mean((im_next_predicted==im_next).float())
        self.predictions = predictions
        self.labels = labels        
        loss = self.loss_func(predictions, labels.reshape(-1, self.act_dim**self.num_dims))
        self.im_next_predicted = im_next_predicted
        self.im_next = im_next
        
        return loss, accuracy

    def plot(self, fp_results='./plots'):
        
        if self.num_dims==2:
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0, 0, ].cpu().numpy())
            axs[0].set_title('Current (' + str(self.seq_sample) + ')')
            axs[0].axis('off')
            axs[1].matshow(self.im_next_predicted[0, 0].cpu().numpy()) 
            axs[1].set_title('Predicted Next (' + str(self.seq_sample) + ')')
            axs[1].axis('off')
            axs[2].matshow(self.im_next[0, 0].cpu().numpy()) 
            axs[2].set_title('True Next (' + str(self.seq_sample) + ')')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true(%s).png'%(fp_results, str(self.seq_sample)))
            #plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0), vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.reshape(-1, self.act_dim, self.act_dim).cpu().numpy(), axis=0), vmin=0, vmax=1) 
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood(%s).png'%(fp_results, str(self.seq_sample)))
            #plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy(%s).png'%(fp_results, str(self.seq_sample)))
            #plt.show()
            plt.close('all')
        
        if self.num_dims==3:
            bi = int(self.im_seq.shape[-1]/2)
            
            #Plot the next images, predicted and true, together
            fig, axs = plt.subplots(1,3)
            axs[0].matshow(self.im_seq[0,0,...,bi].cpu().numpy())
            axs[0].set_title('Current')
            axs[0].axis('off')
            axs[1].matshow(self.im_next[0,0,...,bi].cpu().numpy()) 
            axs[1].set_title('Predicted Next')
            axs[1].axis('off')
            axs[2].matshow(self.im_seq[1,0,...,bi].cpu().numpy()) 
            axs[2].set_title('True Next')
            axs[2].axis('off')
            plt.savefig('%s/sim_vs_true.png'%fp_results)
            plt.show()
            
            #Plot the action distributions, predicted and true, together
            ctr = int((self.act_dim-1)/2)
            pred = self.predictions.reshape(-1, self.act_dim, self.act_dim, self.act_dim).detach().cpu().numpy()
            fig, axs = plt.subplots(1,2)
            p1 = axs[0].matshow(np.mean(pred, axis=0)[...,ctr], vmin=0, vmax=1)
            fig.colorbar(p1, ax=axs[0])
            axs[0].plot(ctr,ctr,marker='x')
            axs[0].set_title('Predicted')
            axs[0].axis('off')
            p2 = axs[1].matshow(np.mean(self.labels.cpu().numpy(), axis=0)[...,ctr], vmin=0, vmax=1) 
            fig.colorbar(p2, ax=axs[1])
            axs[1].plot(ctr,ctr,marker='x')
            axs[1].set_title('True')
            axs[1].axis('off')
            plt.savefig('%s/action_likelihood.png'%fp_results)
            plt.show()
            
            #Plot loss and accuracy
            fig, axs = plt.subplots(1,2)
            axs[0].plot(self.validation_loss, '-*', label='Validation')
            axs[0].plot(self.training_loss, '--*', label='Training')
            axs[0].set_title('Loss')
            axs[0].legend()
            axs[1].plot(self.validation_acc, '-*', label='Validation')
            axs[1].plot(self.training_acc, '--*', label='Training')
            axs[1].set_title('Accuracy')
            axs[1].legend()
            plt.savefig('%s/train_val_loss_accuracy.png'%fp_results)
            plt.show()
            
            plt.close('all')
        
    def save(self, name):
        # self.model.save(name)
        torch.save(self.state_dict(), name)

def train_primme(trainset, n_step, n_samples, test_case_dict, mode = "Multi_Step", num_eps = 25, dims=2, obs_dim=17, act_dim=17, 
                 lr=5e-5, reg=1, pad_mode="circular", pad_value = -1, if_plot=False):
    
    agent = PRIMME(obs_dim=obs_dim, act_dim=act_dim, pad_mode=pad_mode, pad_value=pad_value, learning_rate=lr, 
                   num_dims=dims, mode = mode, epoch = 1, device = device).to(device)
    agent.load_data(h5_paths=trainset, n_step=n_step, n_samples=n_samples)
    for epoch in tqdm(range(1, num_eps+1), desc = 'RL Episodes', leave=True):
        print("epoch:", epoch)
        agent.epoch = epoch
        if agent.epoch % 5 == 0:
            if_plot = True
            agent.subfolder = "pred(%s)_%s_step(%s_%s)_ep(%d)_pad(%s)_md(%d)_sz(%d_%d)_lr(%.0e)_reg(%s)" % (agent.mode, agent.dataset_name, agent.n_step[0], agent.n_step[1], agent.epoch, agent.pad_mode, 
                                                                                                            agent.num_dims, agent.obs_dim, agent.act_dim, agent.learning_rate, agent.reg)
            agent.result_path = ("/").join(['./plots', agent.subfolder])
            if not os.path.exists(agent.result_path):
               os.makedirs(agent.result_path)
        agent.train(if_plot = if_plot)
        if_plot = False
        if agent.epoch % 25 == 0:
            modelname = './data/' + agent.subfolder + '.h5'
            agent.save("%s" % modelname)
            ## Generate test case and Run PRIMME model    
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
                ims_id, fp_primme = run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname)
                sub_folder = "pred" + fp_primme.split("/")[2].split(".")[0].split("pred")[1]
                fs.compute_grain_stats(fp_primme)
                fs.make_videos(fp_primme, sub_folder, ic_shape)
                if grain_shape == "grain":
                    fs.make_time_plots(fp_primme, sub_folder, ic_shape)

def run_primme(ic, ea, miso_array, miso_matrix, nsteps, ic_shape, modelname, pad_mode='circular', pad_value=-1, mode = "Multi_Step", if_plot=False):
    
    # Setup
    agent = PRIMME(pad_mode=pad_mode, pad_value=pad_value, mode = mode, device = device).to(device)
    agent.load_state_dict(torch.load(modelname))
    im = torch.Tensor(ic).unsqueeze(0).unsqueeze(0).float()
    ngrain = len(torch.unique(im))
    tmp = np.array([8,16,32], dtype='uint64')
    dtype = 'uint' + str(tmp[np.sum(ngrain>2**tmp)])
    #miso_array=None
    #if np.all(miso_array==None): miso_array = fs.find_misorientation(ea, mem_max=1) 
    #miso_matrix = fs.miso_conversion(torch.from_numpy(miso_array[None,]))[0]
    fp_save = './data/primme_shape(%s)_%s'% (ic_shape, modelname.split('/')[2])
    # Run simulation
    agent.eval()
    with torch.no_grad():    
        ims_id = im
        for _ in tqdm(range(nsteps), 'Running PRIMME simulation: '):
            _, _, im = agent.step(im.clone().to(device))
            ims_id = torch.cat([ims_id, im.detach().cpu()])
            if if_plot: plt.imshow(im[0,0,].detach().cpu().numpy()); plt.show()
        
    ims_id = ims_id.cpu().numpy()
    
    # Save Simulation
    with h5py.File(fp_save, 'w') as f:
        
        # If file already exists, create another group in the file for this simulaiton
        num_groups = len(f.keys())
        hp_save = 'sim%d'%num_groups
        g = f.create_group(hp_save)
        
        # Save data
        dset = g.create_dataset("ims_id", shape=ims_id.shape, dtype=dtype)
        dset2 = g.create_dataset("euler_angles", shape=ea.shape)
        dset3 = g.create_dataset("miso_array", shape=miso_array.shape)
        dset4 = g.create_dataset("miso_matrix", shape=miso_matrix.shape)
        dset[:] = ims_id
        dset2[:] = ea
        dset3[:] = miso_array #radians (does not save the exact "Miso.txt" file values, which are degrees divided by the cutoff angle)
        dset4[:] = miso_matrix #same values as mis0_array, different format

    #features = fs.compute_features(im, obs_dim=agent.obs_dim, pad_mode=agent.pad_mode, pad_value=agent.pad_value)
    #plt.figure()
    #plt.imshow(im[0, 0].cpu().numpy())

    return ims_id, fp_save






