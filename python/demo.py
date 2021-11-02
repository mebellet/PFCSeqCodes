#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:34:42 2021

@author: Marie E. Bellet
"""
import os
import numpy as np
import preprocessing
animal_id = 'A11'
data_path = '/Volumes/Bellet/Local_Global/data/%s'%animal_id #'path/to/data/%s'%animal_id
out_path = '/Volumes/Bellet/Local_Global/processedData/' #'path/to/processedData/%s'%animal_id

# parameters for psth:
tmin = -0.3 # sec, before each stim
tmax = 1.6 # sec, after each stim
binsize = 0.025 # binsize
sd = 0.5 # standard deviation of gaussian kernel
stepsize = binsize

# create or load data frame containing spikes and meta data
df = preprocessing.getData(data_path, out_path, animal_id, tmin, tmax)

# get psth

psth = preprocessing.getPSTH(df,animal_id,out_path,tmin,tmax,binsize,stepsize,sd=sd)