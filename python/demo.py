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
data_path = 'path/to/data/%s'%animal_id
out_path = 'path/to/processedData/%s'%animal_id

df = preprocessing.getData(data_path, out_path, animal_id)
