# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:15:22 2020

@author: ta3map
"""
import numpy as np
import pickle
import pyspike as spk
import os
import scipy.io
import pandas as pd

#%% neuron numbers 
def GetNeuronNumbers(argument, chosen_neuron_numbers):
    return {
        argument == 2: chosen_neuron_numbers['all_neurons'].tolist(),
        argument == 3: chosen_neuron_numbers['from_stability_test'].dropna().astype(int).tolist()
        }[True]
def GetNeuronNames(argument, chosen_neuron_numbers):
    return {
        argument == 2: chosen_neuron_numbers['all_neurons_names'].tolist(),
        argument == 3: chosen_neuron_numbers['from_stability_test_names'].dropna().tolist()
        }[True]
# choose neurons
# 2 all neurons
# 3 stable

#%% psth from dataset

def LoadDatasetFromMatFiles(main_folder, protocol):
    dataset = []
    number_of_indexes = np.size(protocol, axis=0)
    indexes = list(range(0,number_of_indexes))
        
    for index in indexes:
        matfile = main_folder + protocol['path'][index]
        mat = scipy.io.loadmat(matfile)
        dataset.append(mat)
    return dataset




#%%
def get_psths(type_id, picture_n, neuron, show_n, stimuli_ranges, dataset):
    mat = dataset[neuron]
    stim = stimuli_ranges[type_id][picture_n]
    stimIDs = np.array(mat['spikeData']['stimID'][0].tolist()).flatten()
    found_ids = np.where(stimIDs==stim)
    psth = None
    psth = mat['spikeData']['psth'][0][found_ids][show_n]
    psth = np.squeeze(psth)
    return psth

    
#%%
# Example
#type_id = 5# faceBaby
#picture_n = 2# picture
#neuron = 1
#show_n = 1
#stimuli_ranges = set_A_ranges;
#psth = SpFun.get_psths(type_id, picture_n, neuron, show_n, stimuli_ranges, dataset)

def AllNeuronsPsth(type_id, picture_n, show_n, number_of_indexes,
                stimuli_ranges, dataset, signal_start_time = 0, signal_end_time = 1000):
    indexes = list(range(0,number_of_indexes))
    psths = [];
    signal_time = np.array(list(range(signal_start_time,signal_end_time)))
    silent_neurons = []# neurons without any action    
    for neuron in indexes:
        psth = get_psths(type_id, picture_n, neuron, show_n, stimuli_ranges, dataset)
        spike_times = signal_time[psth == 1]
        if (np.size(spike_times)) == 0:
            spike_times = np.zeros(1)
            silent_neurons.append(neuron)
        psths.append(spike_times)    
    return silent_neurons, psths
# Example
#number_of_neurons = np.size(protocol, axis=0)
#[silent_neurons, psths] = SpFun.AllNeuronsPsth(type_id, picture_n, show_n, number_of_neurons,
#                stimuli_ranges, dataset, signal_start_time = 0, signal_end_time = 1000)

def SaveDatasetSpikeTrains(main_folder, directory, data_folder, dataset, shows_number, 
                              pictures_number, stimuli_ranges, stimuli_names, protocol):
    types_number = len(stimuli_names)
    for type_id in range(0,types_number):
        number_of_neurons = np.size(protocol, axis=0)
        #stim_spike_trains = []
        stim_Silent_neurons = []
        stim_psth = []
        for picture_n in range(0,pictures_number):
            #shows_spike_trains = []
            shows_Silent_neurons = []
            shows_psth = []
            for show_n in range(0,shows_number):
                silent_neurons, psths = AllNeuronsPsth(type_id, picture_n, show_n, number_of_neurons,
                    stimuli_ranges, dataset)
                #edges = [0, np.size(psth)]
                #spike_times = list(range(0, np.size(psth)))
                #spike_times = np.nonzero(np.array(psth, dtype=bool).flatten())
                #spike_train = spk.SpikeTrain(spike_times, edges)
                #shows_spike_trains.append(spike_trains)
                shows_Silent_neurons.append(silent_neurons)
                shows_psth.append(psths)
            #stim_spike_trains.append(shows_spike_trains)
            stim_Silent_neurons.append(shows_Silent_neurons)  
            stim_psth.append(shows_psth)
        
        if not os.path.exists(directory):
            os.mkdir(directory)
        filepath =  directory + '/' + data_folder + '_' + str(type_id) +  '_trains.pickle'
        with open(filepath, 'wb') as f:
            pickle.dump(stim_psth, f)
        sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar
# Example
#dataset_name = 'data_3'
#stimuli_ranges = set_A_ranges;
#stimuli_names = setA_names
#SpFun.save_dataset_spike_trains(main_folder, dataset_name, dataset, shows_number, pictures_number, stimuli_ranges, stimuli_names, protocol)

def LoadSpikeTrain(directory, data_folder, type_id):
    filepath =  directory + '/' + data_folder + '_' + str(type_id) +  '_trains.pickle'
    with open(filepath, 'rb') as f:
        psth = pickle.load(f)
    return psth

def SpikeTrainFromInterval(start_time, end_time, stim_n, show_n, neuron_index, psth):
    data = np.array(psth[stim_n][show_n][neuron_index])
    spike_times = data[(data >= start_time) & (data <= end_time)]
    edges = [0, end_time - start_time]
    silent_neuron = False
    if (np.size(spike_times)) == 0:
            spike_times = np.zeros(1)
            silent_neuron = True
    spike_train = spk.SpikeTrain(spike_times, edges)
    
    return spike_train, silent_neuron

#% all shows all images, one type
def SpikeDirections(folder_in, type_id, start_time, end_time):
    psth = np.asarray(LoadSpikeTrain(folder_in, type_id))
    number_of_neurons = np.size(psth,2)
    neurons = list(range(0,number_of_neurons))
    shows_number = np.size(psth,1)
    stims_number = np.size(psth,0)
    directions_l3 = [];
    for stim_n in range(0,stims_number): # номер картинки (1-15)
        direction_l2 = [];
        for show_n in range(0,shows_number): # номер предъявления (1-30)
            spike_trains_36 = []
            for neuron in neurons:
                st, silent_neuron = SpikeTrainFromInterval(start_time, end_time, stim_n, show_n, neuron, psth)
                spike_trains_36.append(st)
            direction_l1 = spk.spike_directionality_matrix(spike_trains_36, max_tau = 25) #36x36
            direction_l2.append(direction_l1)
        directions_l3.append(direction_l2)
    return directions_l3

#% 15 vs 15 pictures
import sys

def groupMeans(folder_in, group_number, direct, start_time, end_time):
    group_means = []; 
    type_matrix = np.asarray(SpikeDirections(folder_in, group_number, start_time, end_time))
    if direct > 0:
        type_matrix[type_matrix<0] = 0
    elif direct < 0:
        type_matrix[type_matrix>0] = 0         
    pictures_number = np.size(type_matrix, 0)
    for pict_number in range(0,pictures_number):# number of picture
        pict_mean = np.mean(type_matrix[pict_number, :, :, :], 0)
        group_means.append(pict_mean)
        sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar
    return group_means

def groupDiff(folder_in, baby, hide, direct, start_time, end_time):
    baby_means = np.asarray(groupMeans(folder_in, baby, direct, start_time, end_time))
    print(' 1 category is done')
    hide_means = np.asarray(groupMeans(folder_in, hide, direct, start_time, end_time))
    print(' 2 category is done')

    matrix_diff = np.mean(baby_means,0) - np.mean(hide_means,0);
    return(baby_means, hide_means, matrix_diff)
    
#%% graphs
def G_one_picture(picture_mean, weight_on, all_nodes):
    
    positive_direction = np.copy(picture_mean)
    
    positive_direction[positive_direction > 0] = 1;
    positive_direction[positive_direction < 0] = 0;
    
    
    import networkx as nx
    G = nx.DiGraph()
    # add neurons as nodes
    if all_nodes:
        #add all nodes
        for neuron in range(0, np.size(picture_mean, 0)):
            G.add_node(neuron)
    else:
        # add nodes with connections
        for neuron in range(0, np.size(picture_mean, 0)):
            con_profile = np.asarray(positive_direction)[neuron]
            connections = np.argwhere(con_profile == 1)
            if len(connections)>0:
                G.add_node(neuron)
    
    #add edges
    for neuron in range(0, np.size(picture_mean, 0)):
        con_profile = np.asarray(positive_direction)[neuron]
        connections = np.argwhere(con_profile == 1)
        for con in range(0,np.size(connections)):
            neuron2 = int(connections[con])
            weight = picture_mean[neuron, neuron2]
            G.add_edge(neuron, neuron2, weight=weight)

    #% Degrees
    if weight_on:
        weight_string = 'weight'
    else:
        weight_string = 'None'
    nodelist = list(range(0, np.size(picture_mean, 0)))
    indegr = np.asarray(list(G.in_degree(nodelist, weight = weight_string)))[:,1]
    outdegr = np.asarray(list(G.out_degree(nodelist, weight = weight_string)))[:,1]
    eigen_centr = nx.eigenvector_centrality_numpy(G, weight = weight_string, max_iter=100)
    eigen_centr = list(eigen_centr.values())
    return G, indegr, outdegr
#% graphs for all pictures
def degrSet(per_pict_average_values,weight_on, all_nodes):
    IndegrSet = [];
    OutdegrSet = [];
    G_set = [];
    pictures_number = np.size(per_pict_average_values, 0)
    for pict_n in range(0,pictures_number):
        picture_mean = per_pict_average_values[pict_n, :,:]
        G, indegr, outdegr = G_one_picture(picture_mean,weight_on, all_nodes)
        IndegrSet.append(indegr)
        OutdegrSet.append(outdegr)
        G_set.append(G)
    return IndegrSet, OutdegrSet, G_set

#%% statistics



