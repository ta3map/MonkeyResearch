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
#%% load protocol 
def f(data_folder):
    return {
        'data_1': "\protocol\protocol data1.xlsx",
        'data_2': "\protocol\matcha160607-500+500ms.xlsx",
        'data_3': "\protocol\data_3_matcha161122-500+500ms.xlsx"
    }[data_folder]

def load_protocol(data_folder, main_folder):
    pr_path = main_folder + f(data_folder)
    protocol = pd.read_excel (pr_path)
    return protocol

    
#%% psth from dataset

def load_dataset_from_mat_files(main_folder, protocol):
    dataset = []
    number_of_indexes = np.size(protocol, axis=0)
    indexes = list(range(0,number_of_indexes))
        
    for index in indexes:
        matfile = main_folder + protocol['path'][index]
        mat = scipy.io.loadmat(matfile)
        dataset.append(mat)
    return dataset

def get_psths(type_id, picture_n, neuron, show_n, stimuli_ranges, dataset):
    mat = dataset[neuron]
    stim = stimuli_ranges[type_id][picture_n]
    stimIDs = np.array(mat['spikeData']['stimID'][0].tolist()).flatten()
    found_ids = np.where(stimIDs==stim)
    psth = None
    psth = mat['spikeData']['psth'][0][found_ids][show_n]
    psth = np.squeeze(psth)
    return psth

# Example
#type_id = 5# faceBaby
#picture_n = 2# picture
#neuron = 1
#show_n = 1
#stimuli_ranges = set_A_ranges;

#psth = SpFun.get_psths(type_id, picture_n, neuron, show_n, stimuli_ranges, dataset)

def all_neurons_psth(type_id, picture_n, show_n, number_of_indexes,
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
        psths.append(psth)
    
    return silent_neurons, psths

# Example
#number_of_neurons = np.size(protocol, axis=0)

#[silent_neurons, psths] = SpFun.all_neurons_psth(type_id, picture_n, show_n, number_of_neurons,
#                stimuli_ranges, dataset, signal_start_time = 0, signal_end_time = 1000)

def save_dataset_spike_trains(main_folder, dataset_name, dataset, shows_number, 
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
                silent_neurons, psth = all_neurons_psth(type_id, picture_n, show_n, number_of_neurons,
                    stimuli_ranges, dataset)
                #shows_spike_trains.append(spike_trains)
                shows_Silent_neurons.append(silent_neurons)
                shows_psth.append(psth)
            #stim_spike_trains.append(shows_spike_trains)
            stim_Silent_neurons.append(shows_Silent_neurons)  
            stim_psth.append(shows_psth)
        directory = main_folder + '/data/out/spike_trains/' + dataset_name + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)
        filepath =  directory + str(type_id) + '_spike_trains' + '.pickle'
        with open(filepath, 'wb') as f:
            pickle.dump(stim_psth, f)
        sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar

# Example
#dataset_name = 'data_3'
#stimuli_ranges = set_A_ranges;
#stimuli_names = setA_names

#SpFun.save_dataset_spike_trains(main_folder, dataset_name, dataset, shows_number, pictures_number, stimuli_ranges, stimuli_names, protocol)
#%%
def loadSpikeTrain(folder_in, type_id):
    filepath = folder_in + str(type_id) + '_spike_trains' + '.pickle'
    with open(filepath, 'rb') as f:
        psth = pickle.load(f)
    return psth


#%% Low-Level


def spikeTrainFromInterval(start_time, end_time, stim_n, show_n, neuron_index, psth):
    psth_interval = psth[stim_n][show_n][neuron_index][range(start_time,end_time)]
    spike_times = list(range(0, end_time - start_time))
    spike_times = np.array(spike_times)[psth_interval == 1]
    edges = [0, end_time - start_time]
    silent_neuron = False
    if (np.size(spike_times)) == 0:
            spike_times = np.zeros(1)
            silent_neuron = True
    spike_train = spk.SpikeTrain(spike_times, edges)
    
    return spike_train, silent_neuron

#%% all shows all images, one type
def SpikeDirections(folder_in, type_id, start_time, end_time):
    psth = np.asarray(loadSpikeTrain(folder_in, type_id))
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
                st, silent_neuron = spikeTrainFromInterval(start_time, end_time, stim_n, show_n, neuron, psth)
                spike_trains_36.append(st)
            direction_l1 = spk.spike_directionality_matrix(spike_trains_36, max_tau = 25) #36x36
            direction_l2.append(direction_l1)
        directions_l3.append(direction_l2)
    return directions_l3

#%% 15 vs 15 pictures
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
    
#%%  inhibition

def InhiBition(spike_times_1, spike_times_2, tau = 100):
    inhibition_profile = np.array([])
    for i in range(0,np.size(spike_times_1)):
        if np.size(spike_times_2) > 0:
            min_time = spike_times_1[i]
            max_time = spike_times_1[i] + tau
            more = spike_times_2 >= min_time
            less = spike_times_2 <= max_time
            after_first_sp2 = spike_times_1[i] > spike_times_2[0]
            after_last_sp2 = spike_times_1[i] < spike_times_2[-1]
            inhibit_1 = np.logical_and(sum(more & less) == 0, after_first_sp2 & after_last_sp2)
            inhibition_profile = np.append(inhibition_profile, inhibit_1)    
        else:
            inhibition = 0
    inhibition = np.mean(inhibition_profile)
    if np.isnan(inhibition):
        inhibition = 0
    return inhibition, inhibition_profile
#%% inhibit matrix
def inhibitMatrix(spike_times_all, tau = 100):
    inhibit_matrix_1 = np.array([])
    for j in range(0, np.size(spike_times_all, 0)):
        spike_times_1 = spike_times_all[j]
        inhibit_matrix_2 = np.array([])
        for i in range(0, np.size(spike_times_all, 0)):
            spike_times_2 = spike_times_all[i]
            inhibition, inhibition_profile = InhiBition(spike_times_1, spike_times_2, tau = tau)
            inhibit_matrix_2 = np.append(inhibit_matrix_2, inhibition)
        if j == 0:
            inhibit_matrix_1 = inhibit_matrix_2;
        else:
            inhibit_matrix_1 = np.vstack((inhibit_matrix_1, inhibit_matrix_2))
    
    return inhibit_matrix_1

#%% all shows all images, one type
def inhibitOneType(folder_in, type_id, start_time, end_time):
    psth = np.asarray(loadSpikeTrain(folder_in, type_id))
    number_of_neurons = np.size(psth,2)
    neurons = list(range(0,number_of_neurons))
    shows_number = np.size(psth,1)
    stims_number = np.size(psth,0)
    inhibit_l3 = [];
    for stim_n in range(0,stims_number): # номер картинки (1-15)
        inhibit_l2 = [];
        for show_n in range(0,shows_number): # номер предъявления (1-30)
            spike_times_all = []
            sp_all = psth[stim_n][show_n][:]
            signal_time = np.array(list(range(0, np.size(sp_all, 1))))            
            for neuron in neurons:
                spike_time = signal_time[sp_all[neuron,:]==1]
                spike_time = spike_time[spike_time > start_time]
                spike_time = spike_time[spike_time < end_time]
                spike_times_all.append(spike_time)# convert psth into times
                
            inhibit_matrix_1 = inhibitMatrix(spike_times_all, tau = 100)#36x36
            inhibit_l2.append(inhibit_matrix_1)
        inhibit_l3.append(inhibit_l2)
        sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar
    inhibit_l3 = np.asarray(inhibit_l3)        
    return inhibit_l3
