import os
import numpy as np
import pandas as pd
import mne
import time
from mne.viz.utils import _convert_psds
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from xml.etree.ElementTree import parse

def find_close(arr, e):
    low = 0
    high = len(arr) - 1
    idx = -1

    while low <= high:
        mid = int((low + high) / 2)
        if e == arr[mid] or mid == low:
            idx = mid
            break
        elif e > arr[mid]:
            low = mid
        elif e < arr[mid]:
            high = mid
    return idx, idx+1

def auc(x,y,x1,x2):
    if x1 not in x:
        bd1, bd2 = find_close(x, x1)
        x1y = (y[bd2]-y[bd1])/(x[bd2]-x[bd1])*(x1-x[bd1])+y[bd1]
        x = np.insert(x,bd2,x1)
        y = np.insert(y,bd2,x1y)
    if x2 not in x:
        bd1, bd2 = find_close(x, x2)
        x2y = (y[bd2]-y[bd1])/(x[bd2]-x[bd1])*(x2-x[bd1])+y[bd1]
        x = np.insert(x,bd2,x2)
        y = np.insert(y,bd2,x2y)
    idx1 = int(np.where(x==x1)[0])
    idx2 = int(np.where(x==x2)[0])
    ans = np.trapz(y=y[idx1:idx2+1],x=x[idx1:idx2+1])
    return ans

picks_front = ['Fp1','Fp2','Fz','F3','F4','F7','F8','F11','F12']
picks_top = ['Cz','FC3','FCz','FC4','C3','C4','CP3','CPz','CP4']
picks_left = ['FT11','T7','M1','C3','FC3','CP3']
picks_right = ['FT12','T8','M2','C4','FC4','CP4']
picks_back = ['Oz','O1','O2','Pz','P3','P4','P7','P8']
picks_list = list([picks_front, picks_top, picks_left, picks_right, picks_back])
#被试数量
path = 'data\\'
filename = os.listdir(path)
filelist=[]
for i in filename:
    if i[-3:] == 'fif':
        filelist.append(os.path.join(path,i))
print('总计被试数量: ', len(filelist))
all_data = np.empty((1,62),dtype='<U32')

for sub in filelist:
    epochs = mne.read_epochs(sub)
    epochs = epochs.drop_channels(['VEOG'])
    epochs = epochs.apply_baseline(baseline=(None,0))
    epochs_res = epochs.resample(200)
    epochs_res = epochs_res.filter(l_freq=None,h_freq=30)

    #试次数量
    arr = epochs_res.get_data()
    trials = arr.shape[0]
    epo_data = np.empty((1,62),dtype='<U32')
    for i in range(trials):
        condition = list(epochs_res[i].event_id.keys())[0]
        reslist = list()
        result = np.empty((1,62),dtype='<U32')
        j = 0
        for pick in picks_list:
            data = epochs_res[i].get_data(units='uV', picks=pick)
            data = np.squeeze(data)
            data = np.average(data,axis=0)
            psd, freqs = mne.time_frequency.psd_multitaper(epochs[i], picks=pick, fmin=0 , fmax=100, proj=False, n_jobs=1, bandwidth=None, adaptive=False, low_bias=True, normalization='length')
            psd = 10 * np.log10(np.mean(np.mean(psd, axis=0),axis=0) * 1e12)
            #计算不同频段的PSD
            #Delta波1-3Hz
            Delta = auc(freqs,psd,1,3)
            #Theta波4-7Hz
            Theta = auc(freqs,psd,4,7)
            #Alpha波8-11Hz
            Alpha = auc(freqs,psd,8,11)
            #Beta1波12-20Hz
            Beta1 = auc(freqs,psd,12,20)
            #Beta2波21-29Hz
            Beta2 = auc(freqs,psd,21,29)
            #Gamma1波30-65Hz
            Gamma1 = auc(freqs,psd,30,65)
            #Gamma2波66-100Hz
            Gamma2 = auc(freqs,psd,66,100)
            reslist[j:j+12] = np.array([Delta, Theta, Alpha, Beta1, Beta2, Gamma1, Gamma2, np.mean(data[216:220]),np.mean(data[220:226]), np.mean(data[250:260]), np.mean(data[255:265]), np.mean(data[275:285])])
            j += 12

        result[0][:2] = [sub[-12:-8], condition]
        result[0][2:] = reslist
        epo_data = np.concatenate((epo_data,result),axis=0)
    epo_data = epo_data[1:]
    all_data = np.concatenate((all_data,epo_data),axis=0)
    print('当前被试:%s'%(sub[-12:-8]))

df = pd.DataFrame(all_data,index=None)
df.to_excel('eegdata_72var.xlsx')

