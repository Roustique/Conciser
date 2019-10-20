#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import scipy.io.wavfile as wf
import os

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

parser = argparse.ArgumentParser(description='A software that makes lecture recordings more concise.')
parser.add_argument('inputfiles', type=str, nargs='+', help='input .wav file(s)')
args = parser.parse_args()

acc_rate = 1.25
acc_rate0 = 1.0

def clusterize_mask(arr, smooth):
    arr_out = np.zeros(np.size(arr)).astype(bool)
    for i in np.arange(smooth, np.size(arr)-smooth, 2*smooth):
        hassound = np.sum(arr[i-smooth:i+smooth]) > 0
        arr_out[i-smooth:i+smooth].fill(hassound)
    return arr_out
            
content = np.empty((0,2), dtype=np.int16)

for filename in args.inputfiles:
    rate, content_read = wf.read(filename)
    if np.size(np.shape(content_read)) == 1:
        content_read = np.column_stack((content_read, content_read))
    content = np.append(content, content_read, axis=0)
    
del content_read
threschold = np.mean(np.abs(content))
#signalpos = np.where(np.abs(content) > threschold)

#threschold = np.mean(np.abs(content[signalpos]))/10
signalpos_bool = np.abs(content) > threschold
signalpos_merged = np.logical_or(signalpos_bool[:,0], signalpos_bool[:,1])
del signalpos_bool
signalpos_smoothed = clusterize_mask(signalpos_merged, int(rate/20))
del signalpos_merged
signalborders = np.where(signalpos_smoothed[:np.size(signalpos_smoothed)-1] != signalpos_smoothed[1:])[0]
del signalpos_smoothed
#signalpos = np.where(np.abs(content) > threschold)
#signalpos_norepeat = np.unique(signalpos[0])
#signalpos_norepeat = np.append(signalpos_norepeat, np.shape(content)[0])
#signalborders = signalpos_norepeat[np.gradient(signalpos_norepeat) > 2000]
signalborders = np.insert(signalborders, 0, 0)
signalborders = np.append(signalborders, np.size(content[:,0]))

newcontent = np.empty((0,2), dtype=np.int16)

for i in (np.arange(1, np.size(signalborders))):
    if np.mean(np.abs(content[signalborders[i-1]:signalborders[i],:])) > threschold:
        lborder = int(np.max([signalborders[i-1]-rate/15, 0]))
        uborder = int(np.min([signalborders[i]+rate/15, np.size(content[:,0])]))
        acc_size = int(np.floor((uborder-lborder)/acc_rate0))
        acc_part = np.empty((acc_size,2))
        nonacc_part = content[lborder:uborder,:]
        acc_part = nonacc_part[np.floor(np.arange(acc_size) * acc_rate0).astype(int),:]
        newcontent = np.append(newcontent, acc_part, axis=0)
        
wf.write('output_temp.wav', rate, newcontent)

with WavReader('output_temp.wav') as reader:
    with WavWriter('output.wav', reader.channels, reader.samplerate) as writer:
        tsm = phasevocoder(reader.channels, speed=acc_rate)
        tsm.run(reader, writer)
        
os.remove('output_temp.wav')
        
