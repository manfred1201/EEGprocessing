#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from xml.etree.ElementTree import parse
xml = parse('rawdata\mymontage.xml')
root = xml.getroot()
mydic = dict()
for channel in root.findall('CHANNEL'):
    label = channel.find('LABEL')
    pos = channel.find('POSITION')
    if pos is None:
        continue
    lis = pos.text.split()
    a = np.array(lis).astype(np.float64) / 1000
    a[0] = -a[0]
    a[1] = -a[1]
    mydic[label.text] = a
mymontage = mne.channels.make_dig_montage(ch_pos=mydic,coord_frame='head')


#导入数据并剔除无用channels
raw = mne.io.read_raw_eeglab('rawdata/eeg/f008.set')
raw.set_channel_types({'VEOG':'eog'})
raw.pick(picks='all',exclude=['HEOG','EKG','EMG','Trigger'])

#电极定位
raw.set_montage(mymontage,on_missing='warn')

#提取社会奖赏实验的片段
events = mne.events_from_annotations(raw)
ind = np.where(events[0][:,2] == events[1]['14'])[0].min()
start_t = events[0][ind,0] - 5000
ind = np.where(events[0][:,2] == events[1]['18'])[0].max()
end_t = events[0][ind,0] + 8500
raw_cropped = raw.copy()
raw_cropped.crop(tmin=start_t/1000,tmax=end_t/1000)


#插值坏导
raw_cropped.plot(start=20,duration=1,n_channels=33,block=True,title='请检查并选中坏导')   #定义坏导
plt.show()
badflag = False
if raw_cropped.info['bads']:
    print('已选择坏导: ',raw_cropped.info['bads'], '开始进行插值')
    badflag = True
else:
    print('无坏导，跳过插值')
if badflag:
    raw_cropped.load_data()
    raw_cropped.interpolate_bads(exclude=['F11','F12','FT11','FT12'])
    raw_cropped.plot(start=20,duration=1,n_channels=33,block=True,title='坏导插值完成，如无误请关闭窗口')
    plt.show()



#重参考
raw_ref = raw_cropped.copy()
raw_ref.load_data()
raw_ref.set_eeg_reference(ref_channels=['M1','M2'])
raw_ref.plot(start=20,duration=1,block=True,title='重参考完成，无误请关闭窗口')
plt.show()


#滤波
raw_filter = raw_ref.copy()
raw_filter.filter(l_freq=1,h_freq=30)
raw_filter.notch_filter(freqs=50)
raw_filter.plot_psd(fmax=60)
plt.show(block=False)
raw_filter.plot(start=20,duration=1,block=True,title='滤波完成，准备ICA，无误请关闭窗口')

#ICA
ica = mne.preprocessing.ICA(n_components=10, method='picard', max_iter=800)
ica.fit(raw_filter)
raw_filter.load_data()
ica.plot_components()
ica.plot_sources(raw_filter, show_scrollbars=False, title='请选择需要去除的成分')
plt.show(block=True)
print(ica)
raw_recons = raw_filter.copy()
raw_recons = ica.apply(raw_recons)
raw_filter.plot(start=20,duration=1,n_channels=33,title='ICA处理前, 确认请关闭')
raw_recons.plot(start=20,duration=1,n_channels=33,title='ICA处理后, 确认请关闭')
plt.show(block=True)

#提取epochs
events = mne.events_from_annotations(raw_recons)
event_dic = {'pos' : events[1]['20'], 'neg' : events[1]['22']}
reject_criteria = dict(eeg=100e-6)  # 100 µV
epochs = mne.Epochs(raw_recons, events[0], event_id=event_dic, preload=True, tmax=1, tmin=-0.2, reject=reject_criteria)
epochs.plot(events=events[0],block=True,title='请目视挑选出坏EPOCHES')
plt.show()

#对两个condition的epochs做平均
#epochs.equalize_event_counts(['pos','neg']) 是否需要将两个condition一一对应？？
pos_epochs = epochs['pos']
neg_epochs = epochs['neg']

pos_evoked = pos_epochs.average()
neg_evoked = neg_epochs.average()
neg_evoked.plot()
plt.show(block=True)
pos_evoked.save('pos_ave.fif')
neg_evoked.save('neg_ave.fif')
#最终可视化差异
mne.viz.plot_compare_evokeds(dict(positive=pos_evoked,negative=neg_evoked),combine='mean',picks = np.arange(32), invert_y=True, legend=True, ci=True)
plt.show()