# pre-sets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


main_folder = 'F:\YandexDisk\MonkeysDatas'
import os
os.chdir(main_folder + '/analysis')
import SpFun

data_folder = 'data_1'

# read protocol
filepath = main_folder + '/protocol/' + data_folder + '.xlsx'
protocol = pd.read_excel(filepath, sheet_name=0)
set_properties = pd.read_excel(filepath, sheet_name=1)
chosen_neuron_numbers = pd.read_excel(filepath, sheet_name=2)
neuron_names = protocol['neuron_name'].tolist()


# choose neurons
# 2 for all neurons
# 3 stable

chosen_neurons = SpFun.GetNeuronNumbers(3, chosen_neuron_numbers)
chosen_neuron_names = SpFun.GetNeuronNames(3, chosen_neuron_numbers)
chosen_neurons_number = len(chosen_neurons)

#% setA stimuli names and ranges
set_names = set_properties['set names'].tolist()
shows_number = int(set_properties['shows number'][0])
pictures_number = int(set_properties['pictures number'][0])
stimuli_number = len(set_names)
neurons_number = len(protocol)


#%% convert dataset mat files into spike-trains(psths)

set_ranges = [];
for s in range(1,stimuli_number+1):
    set_ranges.append(list(range(1+pictures_number*(s-1), 1+pictures_number*s)))
    
# Whole dataset from mat files
dataset = SpFun.LoadDatasetFromMatFiles(main_folder + '/data', protocol)
#%%
n = 0
mat = dataset[n]
c = 0# category
p = 0# picture
stim = set_ranges[c][p]
stimIDs = np.array(mat['spikeData']['stimID'][0].tolist()).flatten()
found_ids = np.where(stimIDs==stim)
s = 1 # show
spikeTrain = mat['spikeData']['spikeTrain'][0][found_ids][s]
#%% SpikeTrains from dataset

def getMatData(dataset, data_name, stimuli_number, pictures_number, shows_number):
    data_n = []# neurons(1, 2, 3 ...)
    for n in range(len(dataset)):
        mat = dataset[n]
        data_c = []# category (baby, grooming ...)
        for c in range(stimuli_number):
            data_p = []# picture number (1, 2, 3 ...)
            for p in range(pictures_number):
                stim = set_ranges[c][p]
                stimIDs = np.array(mat['spikeData']['stimID'][0].tolist()).flatten()
                found_ids = np.where(stimIDs==stim)
                data_s = []# shows (1, 2, 3 ...)
                for s in range(shows_number):
                    data = None
                    data = mat['spikeData'][data_name][0][found_ids][s]
                    data = np.squeeze(data)
                    data_s.append(data)
                data_p.append(data_s)
            data_c.append(data_p)
        data_n.append(data_c)
    return data_n

spikeTrains = getMatData(dataset, 'spikeTrain', stimuli_number, pictures_number)