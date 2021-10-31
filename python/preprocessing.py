#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:41:36 2021

@author: Marie E. Bellet

Preprocessing of spike data
Generates pandas dataframe containing aligned spike times and meta data
Generates PSTH

"""

# imports
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy import signal
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn import mixture#
from glob import glob as glob
import os
from neo import io
from scipy.ndimage import gaussian_filter1d
import h5py

def getData(data_path,out_path,animal_id):
    '''
    Data preprocessing function
    loads spike times and meta data
    alignment of spikes to stimulus onsets
    '''
    sf = 30000; # sampling frequency
    tmin = -.4 #s, before each stim
    tmax = 1.6 #s, after each stim
    
    dates = os.listdir(data_path)
    dates = np.sort([dates[i] for i in range(len(dates)) if dates[i][0]=='2']) # filter
    print(dates)

    folders = [os.path.join(data_path,'%s'%date) for date in dates]
    
    df = pd.DataFrame({'PFC_MU': [], # Spike time relative to stimulus onset
                       'PFC_SU': [],
                       'TrialID':[], 
                       'ItemID':[], # position of stimulus in a trial
                      'StimID':[],# ID of stimulus being displayed
                      'StimName':[], # Name of the image
                      'StimOn':[],
                      'blockID': [],
                      'blockType':[],
                      'date': [],
                      'StimDur':[],
                      'ISIDur':[]}) # ID of the block of trials
    
    # list with stim IDs per session, located in data directory
    stim_list = pd.read_excel(os.path.join(data_path,'stim.xlsx'),header=0)
    
    for sesID,folder in enumerate(folders):
        date = dates[sesID]
        # check if session dataframe exists
        if os.path.exists(os.path.join(out_path,'%s_stims_spikes_dataframe_%s.pkl'%(animal_id,date))):
            print('%s: load existing data frame'%date)
            df_single = pd.read_pickle(os.path.join(out_path,'%s_stims_spikes_dataframe_%s.pkl'%(animal_id,date)))
        else:
            print('%s: align spikes and load metadata'%date)
            # create dataframe for single session
            df_single = pd.DataFrame({'PFC_MU': [], # Spike time relative to stimulus onset
                           'PFC_SU': [],
                          'TrialID':[], 
                          'ItemID':[], # position of stimulus in a trial
                          'StimID':[],# ID of stimulus being displayed
                          'StimName':[], # Name of the image
                          'StimOn':[],
                          'blockID': [],
                          'blockType':[],
                          'date': [],
                          'StimDur':[],
                          'ISIDur':[]}) # ID of the block of trials
            times,IDs,blockID,blockType,trialID,itemID,reward_time,stim_dur,isi_dur = Match_stimuli_ID_to_photodiodes(folder,dates[sesID],out_path)
            time_between_stim = (float(stim_dur)+float(isi_dur))/1000 # SOA in sec
            print('SOA:',time_between_stim)
            
            # get stim ID form session
            row = np.where(stim_list['date']==float(dates[sesID]))[0]
            stimA = np.array(stim_list['A'].iloc[row])[0]
            stimB = np.array(stim_list['B'].iloc[row])[0]
            stim_names = [stimA,stimB]
            print('Stimuli:',stimA,stimB)
            
            # get spikes
            PFC_MU,PFC_SU,ch_num_pfc = getSpikes(folder)
        
            # align spikes
            # Multi-unit activity
            for ch in range(len(PFC_MU)):
                rowcount = 0
                spikes = PFC_MU[ch]
                if len(spikes)>0: # it might be that there is no MU in the channel
                    spikes /= sf # convert to s
                    
                for s,ph in enumerate(times): #loop through stim in trial
                    if len(spikes)>0:
                        sp = spikes[((spikes-ph)>=tmin) & ((spikes-ph)<tmax)] - ph
                    else:
                        sp = []
                    #add to dataframe
                    if ch==0: # Before all, add to the dataframe the info about stimuli
                        df_single = df_single.append({'PFC_MU':[],
                                        'PFC_SU':[],
                                  'TrialID':trialID[s],
                                  'ItemID':itemID[s], # position of stimulus in a trial
                                  'StimOn':times[s],
                                  'RewardOn':reward_time[s],
                                  'StimID':IDs[s], # id 0 or 1
                                  'StimName': stim_names[IDs[s]], # image name
                                  'blockID': blockID[s], # block ID 0-3
                                  'blockType':blockType[s], # block type 0 or 1 (xx or xY)
                                  'date': dates[sesID],
                                  'StimDur':stim_dur,
                                  'ISIDur':isi_dur},ignore_index=True)
                    df_single['PFC_MU'].iloc[rowcount].append(sp)
                    rowcount += 1
                    
            # loop through single units, PFC
            for ch in range(len(PFC_SU)):
                rowcount = 0
                spikes = PFC_SU[ch]
                spikes /= sf # convert to s
                for s,ph in enumerate(times): #loop through stim in trial
                    sp = spikes[((spikes-ph)>=tmin) & ((spikes-ph)<tmax)] - ph
                    
                    df_single['PFC_SU'].iloc[rowcount].append(sp)
                    rowcount += 1
                    
         
            # store data frame from single session
            df_single.to_pickle(os.path.join(out_path,'%s_stims_spikes_dataframe_%s.pkl'%(animal_id,date))) # store the dataframe
            np.save(os.path.join(out_path,'%s_SU_channel_numbers_PFC_%s.npy'%(animal_id,date)),ch_num_pfc)
        # append dataframe from single session to total dataframe
        frames = [df,df_single]
        df = pd.concat(frames,ignore_index=True,sort=False)
        
    df.to_pickle(os.path.join(out_path,'%s_stims_spikes_dataframe.pkl'%animal_id)) # store the dataframe

    return df

def getPSTH(df_path, animal_id):
    
    df = pd.read_pickle(os.path.join(df_path,'%s_stims_spikes_dataframe_%s.pkl'%(animal_id)))
    
    dates = np.unique(df.date)
    ndates = len(dates)
    nitems = np.max(df.ItemID)
    stimID = np.unique(df.StimID)
    nstim = len(stimID)
    print(list(df.keys()))
    print('Recording dates:',dates)

def rate_binning(spike_times,time_bins,binsize):
    average = np.zeros((len(spike_times),len(time_bins)))
    for i,t in enumerate(time_bins):
        
        for chan in range(len(spike_times)):
            include = (spike_times[chan]>=t) & (spike_times[chan]<(t+binsize))
            average[chan,i] = sum(include)/binsize
    return average


def smoothing(signal,sd,binsize):
    ''' aplly gaussian filter per trial'''
    kernel = gaussian(signal.shape[-1],sd/binsize)
    ga = np.zeros(signal.shape)
    if len(signal.shape)>3:
        for dim1 in range(signal.shape[0]):
            for dim2 in range(signal.shape[1]):
                for dim3 in range(signal.shape[2]):
                    ga[dim1,dim2,dim3,:] = filters.convolve1d(signal[dim1,dim2,dim3,:], kernel/kernel.sum())
    else:
        for dim1 in range(signal.shape[0]):
            for dim2 in range(signal.shape[1]):
                ga[dim1,dim2,:] = filters.convolve1d(signal[dim1,dim2,:], kernel/kernel.sum())
                
    return ga


def get_stim_time_and_ID(path,trial_count):
    
    blocks = [[1,1,2],[1,2,1],[2,2,1],[2,1,2]] # this is how block types are coded in matlab script
    stimIDs = [[[0,0,0,0],[0,0,0,1]],
               [[0,0,0,1],[0,0,0,0]],
               [[1,1,1,1],[1,1,1,0]],
               [[1,1,1,0],[1,1,1,1]]]
    log_file = sio.loadmat(path)
    
    Stim_times = log_file['Log'][0,0]['Trial']['StimOnsetTimes'][0]
    stim_dur = log_file['Stm']['ElementDuration'][0,0][0][0]
    isi_dur = log_file['Stm']['InterElementInterval'][0,0][0][0]
    
    # get block
    seq = log_file['Stm'][0,0]['SequenceElements'][0]
    seq_comp = [sum(blocks[i]==seq) for i in range(len(blocks))]
    block = np.where(np.array(seq_comp)==len(blocks[0]))[0][0]
    
    # deviants
    deviant = [log_file['Log'][0,0]['Trial']['TrialType'][0,i][0,0] for i in range(len(log_file['Log'][0,0]['Trial']['TrialType'][0,:]))]
    Onsets = []
    IDs = []
    Trial_count = []
    Item_count = []
    for i in range(len(Stim_times)):
        for j in range(len(Stim_times[i])):
            if Stim_times[i][j,0]!=0:
                Onsets.append(Stim_times[i][j,0])
                IDs.append(stimIDs[block][deviant[i]][j])
                Trial_count.append(trial_count)
                Item_count.append(j)
        trial_count += 1
    df = pd.DataFrame({'Onsets': Onsets,
                      'IDs':IDs,
                      'Trial_count':Trial_count,
                      'Block':np.repeat(block,len(Onsets)),
                      'StimDur':np.repeat(stim_dur,len(Onsets)),
                      'ISIDur':np.repeat(isi_dur,len(Onsets)),
                      'Item_count':Item_count})
    return(df,log_file,trial_count)
    
    
def Match_stimuli_ID_to_photodiodes(folder,date,out_path):
        
    # concatenate all log files in one panda dataframe
    all_log_path = np.sort(glob(os.path.join(folder,'*Log_*')))
    dfs = []
    log_files = []
    trial_count = 0 # keep track of total trial number, initialize with 0
    for f in all_log_path:
        df,log_file,trial_count = get_stim_time_and_ID(f,trial_count)
        dfs.append(df) 
        log_files.append(log_file)

    # load analog signal
    
    file = glob(folder+'/*.nev')        
    reader = io.BlackrockIO(file[0])
    seg = reader.read_segment(lazy=True)
    events = np.asarray(seg.events[0].load().labels)
    timestamps = np.asarray(seg.events[0].load().times)
    #TTL1 = timestamps[events == b'1']
    reward_times = timestamps[events == b'8']
    stimon_times = timestamps[events == b'4']
    blockon_times = timestamps[events == b'1']
    # count number of stim onsets within each block and log file
    stim_per_block = []
    for i in range(len(blockon_times)):
        if i==len(blockon_times)-1:
            stim_per_block.append(sum(stimon_times>blockon_times[i]))
        else:
            stim_per_block.append(sum((stimon_times>blockon_times[i]) & (stimon_times<blockon_times[i+1])))
    stim_per_logfile = [len(np.unique(dfs[block]['Trial_count'])) for block in range(len(dfs))]
                        
    print('N blocks:',len(all_log_path),'TTL block onset events:',len(blockon_times))
    print(stim_per_block)
    print(stim_per_logfile)

    # find match of TTL blocks and log files:
    if (len(stim_per_block)!=len(stim_per_logfile)) | (np.sum(np.array(stim_per_block)==np.array(stim_per_logfile))!=len(stim_per_block)):
        match = np.zeros(len(all_log_path)).astype(int)
        for i,n in enumerate(stim_per_logfile):
            match[i] = np.where(n==np.array(stim_per_block))[0]
        blockon_included = blockon_times[match]
    else:
        blockon_included = blockon_times
    
    sortind = np.argsort(blockon_included)
    dfs2 = []
    for ii in sortind:
        dfs2.append(dfs[ii])
    dfs = dfs2
    del dfs2
    blockon_included = blockon_included[sortind]
    
    # Photodiode times
    file = glob(folder+'/*.ns3')        
    reader = io.BlackrockIO(file[0])
    seg = reader.read_segment(lazy=True)
    photo_fs = 2000 # in Hz. Sampling frequency of the photodiode
    stim_field = 3
    gray_field = 4
    
    photo_stim = np.asarray(seg.analogsignals[0].load(time_slice=None, channel_indexes=[stim_field])).squeeze()
    threshold_cross = (photo_stim-np.median(photo_stim))>(10*np.median(abs(photo_stim-np.median(photo_stim)))/0.6745)

    photo_stim_onset = np.where(np.logical_and(threshold_cross[1:],threshold_cross[:-1]==False))[0]
    # catch: if no onset detected in this channel, use different analog channel:
    if len(photo_stim_onset)==0:
        stim_field = 4
    
        photo_stim = np.asarray(seg.analogsignals[0].load(time_slice=None, channel_indexes=[stim_field])).squeeze()
        
        threshold_cross = (photo_stim-np.median(photo_stim))>(10*np.median(abs(photo_stim-np.median(photo_stim)))/0.6745)
        photo_stim_onset = np.where(np.logical_and(threshold_cross[1:],threshold_cross[:-1]==False))[0]

    # only keep photodiode times that are intentionally triggered and not those induced by changes in overall screen luminosity
    photo_stim_high = np.zeros(len(photo_stim_onset)*int(photo_fs*.05))
    for i,t in enumerate(photo_stim_onset):
        photo_stim_high[i*int(photo_fs*.05):(i+1)*int(photo_fs*.05)] = photo_stim[t:t+int(photo_fs*.05)]
    photo_stim_high=photo_stim_high>.9*np.median(photo_stim_high)
    include = np.ones(len(photo_stim_onset)).astype(bool)
    for i,t in enumerate(photo_stim_onset):
        if not(any(photo_stim_high[i*int(photo_fs*.05):(i+1)*int(photo_fs*.05)])):
            include[i] = False
    photo_stim_high = None
    photo_stim_onset = photo_stim_onset[include]/photo_fs
    
    stim_time = []
    reward_time = []
    stim_IDs = []
    blockIDincr = 0
    blockID = []
    blockType = []
    trialID = []
    itemID = []
    for block in range(len(dfs)):
        
        if len(np.diff(dfs[block]['Onsets']))>0:
            which_block = dfs[block]['Block'][0] # get block ID
            trials = dfs[block]['Trial_count'] # trial numbers
            items = dfs[block]['Item_count'] # item numbers 0-3
            ids = dfs[block]['IDs'] # stim IDs
            if block==len(dfs)-1:
                matching_onset = (photo_stim_onset>=blockon_included[block])
            else:
                matching_onset = (photo_stim_onset>=blockon_included[block]) & (photo_stim_onset<blockon_included[block+1])
            print('Block:%s, photo onsets: %s, log file onsets: %s'%(block,sum(matching_onset),len(dfs[block]['Onsets'])))
            log_onsets = dfs[block]['Onsets']
            # if the number of photodiode times matches with the log file stim onsets, everything is fine
            matched = False
            while matched==False:
                n_diff = sum(matching_onset) - len(log_onsets)
                block_photo = np.diff(photo_stim_onset[matching_onset]) # difference of photodiode time onsets
                if len(log_onsets)>sum(matching_onset):
                    block_log = np.diff(log_onsets[:n_diff])
                else:
                    block_log = np.diff(log_onsets)
                matching = block_photo-block_log
                plt.plot(matching)
                plt.show()
                if sum(matching_onset) == len(log_onsets): #check that there are as many photo onset in block as in log file
                    # get rewardtime corresponding to stimulus
                    reward_matched = []
                    for j,s in enumerate(photo_stim_onset[matching_onset]):
                        if j<len(photo_stim_onset[matching_onset])-1:
                            ii = np.where((reward_times>s) & (reward_times<photo_stim_onset[matching_onset][j+1]))[0]
                            if len(ii)>0:
                                reward_matched.append(reward_times[ii][0])
                            else:
                                reward_matched.append(np.nan)
                        else:
                            ii = np.argmin(abs((reward_times-s)))
                            reward_matched.append(reward_times[ii])
                    reward_time.append(reward_matched)
                    print('Reward delay:',np.nanmean(reward_matched-photo_stim_onset[matching_onset]))
                    stim_time.append(photo_stim_onset[matching_onset])
                    stim_IDs.append(ids)
                    blockID.append(np.repeat(blockIDincr,sum(matching_onset)))
                    blockType.append(np.repeat(which_block,sum(matching_onset)))
                    trialID.append(trials)
                    itemID.append(items)
                    blockIDincr += 1
                    matched = True
            
                # otherwise, each trial has to be checked individually:
                else:
                    
                    out = np.where(abs(matching)>.05)[0][0]
                    log_onsets = np.delete(np.array(log_onsets),out+1)
                    trials = np.delete(np.array(trials),out+1)
                    items = np.delete(np.array(items),out+1)
                    ids = np.delete(np.array(ids),out+1)
                    
        stim_IDs = np.concatenate(stim_IDs)
        stim_time = np.concatenate(stim_time)
        reward_time = np.concatenate(reward_time)
        blockID = np.concatenate(blockID)
        blockType = np.concatenate(blockType)
        stim_dur = dfs[0]['StimDur'][0]
        isi_dur = dfs[0]['ISIDur'][0]
        trialID = np.concatenate(trialID)
        itemID = np.concatenate(itemID)
   
    return(stim_time,stim_IDs,blockID,blockType,trialID,itemID,reward_time,stim_dur,isi_dur)
    
    
# spike alignment functions
def getSpikes(folder):
    '''
    Inputs: folder (directory of data)
    '''
    MUSpikes = []
    SUSpikes = []
    ch_number = []
    area = 'PFC'
    # check if single units exist:
    if os.path.exists(os.path.join(folder,area,'clusters_1.csv')):
        # use function that separates multi-units and single units
        MUSpikes,SUSpikes,ch_number = getSortedSpikes(folder,area)
    else:
        # check if spikes.mat exists:
        if os.path.exists(os.path.join(folder,area,'%sSpikes.mat'%area)):
            # only get multi-units otherwise
            # import spike stimes from Fanis' spike extraction method
            f = h5py.File(os.path.join(folder,area,'%sSpikes.mat'%area))
            spikes = f['spikes%s'%area]
            nch = spikes.shape[0] # number of channels
            for ch in range(nch):
                sp = f[spikes[ch,0]]['times'][:,0]
                w = f[spikes[ch,0]]['waveforms']
                MUSpikes.append(sp[w[14,:]<0]) # time at 30 kHz # keep positive only!

    return MUSpikes,SUSpikes,ch_number

def getSortedSpikes(folder): # function in case of spike sorting
    MUSpikes = [] # multi unit activity per channel
    SUSpikes = [] # single unit activity per channel
    ch_number = []
    area = 'PFC'
    # import spike times from Fanis' spike extraction method
    f = h5py.File(os.path.join(folder,area,'%sSpikes.mat'%area))
    spikes = f['spikes%s'%area]
    nch = spikes.shape[0] # number of channels
    for ch in range(nch):
        sp = f[spikes[ch,0]]['times'][:,0] # time at 30 kHz
        clusters = np.loadtxt(os.path.join(folder,area,'clusters_%s.csv'%(ch+1)),delimiter=',')[1:] # discard first entry
        clu_val = np.unique(clusters)
        clu_val = clu_val[clu_val>0] # discard cluster 0 = junk
        if np.min(clu_val)!=1:
            MUSpikes.append([]) # append empty entry if no MUA in this channel
        for c in clu_val:
            if c == 1: # cluster 1 = MUA
                MUSpikes.append(sp[clusters==c])
            else: # others are single units
                SUSpikes.append(sp[clusters==c])
                ch_number.append(ch)

    return MUSpikes,SUSpikes,ch_number
