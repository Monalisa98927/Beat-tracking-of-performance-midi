import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.utils import data
from scipy import interpolate
import numpy as np

music_ids = []
current_id = ''
time_vs_beat_dic = {} # { '2298': {'time': [], 'beat_0_to_8': []}}
model_input_dic = {}

current_time = 1000000
current_beat = 1000000

with open('time_vs_beat.txt') as file:
    for line in file:
        if line[0:2] == 'id':
            music_id = line[5:9]
            music_ids.append(music_id)

            current_id = music_id
            time_vs_beat_dic[current_id] = {'time': [], 'beat_0_to_8': [], 'beat': []}

        else:
            time = int(line.replace('\n','').split(' ')[0])
            beat = float(line.replace('\n','').split(' ')[1])
            
            if not (time == current_time and beat == current_beat):
                time_vs_beat_dic[current_id]['time'].append(time)
                time_vs_beat_dic[current_id]['beat_0_to_8'].append(beat)
                current_time = time
                current_beat = beat

for id in music_ids:
    time_list = time_vs_beat_dic[id]['time']
    beat_list = time_vs_beat_dic[id]['beat_0_to_8']
    model_input_dic[id] = {}

    current_beat = 0
    count_8_beat = 0
    for t, b in zip(time_list, beat_list):
        delta = b - current_beat
        if delta < -3.5:
            count_8_beat += 1
        current_beat = b
        time_vs_beat_dic[id]['beat'].append(b+count_8_beat*8)
        
        try:
            model_input_dic[id][b+count_8_beat*8].append(t)
        except:
            model_input_dic[id][b+count_8_beat*8] = []
            model_input_dic[id][b+count_8_beat*8].append(t)

    # plt.plot(time_vs_beat_dic[id]['time'], time_vs_beat_dic[id]['beat'], label=id)
    
# plt.xlabel('onset time')
# plt.ylabel('onset beat')
# plt.legend()
# plt.savefig('musicnet_time_vs_beat/time_vs_beat.jpg', dpi=3000)

# print('==========================================================================')
# print(model_input_dic)


for id in music_ids:
    x_beat = []
    y_time = []
    print('music id:',id)
    for beat in model_input_dic[id]:
        x_beat.append(beat)
        y_time.append(sum(model_input_dic[id][beat])/len(model_input_dic[id][beat]))

    # get the missing int beat
    max_beat = max(x_beat)
    max_int_beat = math.floor(max_beat)
    all_int_beats = range(max_int_beat)
    missing_int_beats = list(set(all_int_beats) - set(x_beat))
    # print(missing_int_beats)

    interpo_beats = []
    for m in missing_int_beats:
        if m <= min(x_beat) or m >= max(x_beat):
            # print(m)
            pass
        else:
            interpo_beats.append(m)

    p = interpolate.interp1d(x_beat,y_time,kind='slinear')
    interpo_time = p(interpo_beats)

    plt.figure()
    plt.scatter(x_beat, y_time, c='b', s=0.05)
    plt.scatter(interpo_beats, interpo_time, c='r', s=1, marker = 'x')
    plt.savefig(f'musicnet_time_vs_beat/interpolation_{id}.jpg', dpi=3000)

    onset_time_list = list(interpo_time)
    onset_beat_list = list(interpo_beats)
    for beat, time in zip(x_beat, y_time):
        if beat - int(beat) == 0.0:
            onset_time_list.append(time)
            onset_beat_list.append(beat)

    svl_file_interpo = open(f'musicnet_test_set_beats/{id}_interpo.svl','w')
    svl_file = open(f'musicnet_test_set_beats/{id}.svl','w')
    svl_file_interpo.writelines(['<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE sonic-visualiser>\n<sv>\n  <data>\n'])
    svl_file.writelines(['<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE sonic-visualiser>\n<sv>\n  <data>\n'])

    frame_list = []
    for time,beat in zip(onset_time_list,onset_beat_list):
        frame = int(time*44100/1000)
        frame_list.append(frame)

    svl_file.writelines([f'    <model id="1" name="" sampleRate="44100" start="{min(frame_list)}" end="{(max(frame_list)+1)}" type="sparse" dimensions="1" resolution="1" notifyOnAdd="true" dataset="0" />\n    <dataset id="0" dimensions="1">\n'])
    svl_file_interpo.writelines([f'    <model id="4" name="" sampleRate="44100" start="{min(frame_list)}" end="{(max(frame_list)+1)}" type="sparse" dimensions="1" resolution="1" notifyOnAdd="true" dataset="3" />\n    <dataset id="3" dimensions="1">\n'])

    for time,beat in zip(onset_time_list,onset_beat_list):
        frame = int(time*44100/1000)
        if beat in interpo_beats:
            svl_file_interpo.writelines([f'\t<point frame="{frame}" label="{int(beat)}" />\n'])
        else:
            svl_file.writelines([f'\t<point frame="{frame}" label="{int(beat)}" />\n'])

    svl_file.writelines(['    </dataset>\n  </data>\n  <display>\n    <layer id="2" type="timeinstants" name="Time Instants" model="1"  plotStyle="0" colourName="Purple" colour="#c832ff" darkBackground="false"/>\n  </display>\n</sv>'])
    svl_file_interpo.writelines(['    </dataset>\n  </data>\n  <display>\n    <layer id="5" type="timeinstants" name="Time Instants" model="4"  plotStyle="0" colourName="Orange" colour="#ff9632" darkBackground="false"/>\n  </display>\n</sv>'])

    svl_file_interpo.close()
    svl_file.close()