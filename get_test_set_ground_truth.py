import numpy as np
import matplotlib.pyplot as plt
import os
import csv

onset_time = []
onset_beat = []

music_ids = ['2303','2106','2382','2556','2416','2298','1819','2191','2628','1759']
music_len = [701,1939,1909,1380,1329,841,1272,515,1470,1651]
label_path = '/gpfsnyu/scratch/kf2395/musicnet/musicnet/musicnet'

with open('onset_time.txt') as time:
    for i,t in enumerate(time):
        if t.replace('\n','') in music_ids:
            print(t.replace('\n',''), i)
        if t.replace('\n','') not in music_ids:
            onset_time.append(t.replace('\n',''))

with open('onset_beat.txt') as beat:
    for b in beat:
        label = b.split(' ')[0].replace('\n','')
        label = int(label) / 60
        onset_beat.append(label)

test_set_beat_ground_truth = []
for t,b in zip(onset_time,onset_beat):
    item = str(int(t)) + ' ' + str(b) + '\n'
    test_set_beat_ground_truth.append(item)

time_vs_beat_file = open('test_set_beat_ground_truth.txt',mode='w')
time_vs_beat_file.writelines(test_set_beat_ground_truth)
time_vs_beat_file.close()

def load_musicnet_csv(csv_file):
    """
    Loads a CSV file containing the MusicNet data.
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

print('=======================================================')

for i,music_id in enumerate(music_ids):

    file_path = label_path + '/test_labels/' + music_id + '.csv'
    data = load_musicnet_csv(file_path)

    svl_file = open(f'musicnet_test_set_beats/{music_id}_ground_truth.svl','w')
    svl_file.writelines(['<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE sonic-visualiser>\n<sv>\n  <data>\n'])
        
    frame_list = []
    for index,row in enumerate(data[1:]):

        o_t = int(row[0])
        o_b = float(row[4])
        frame = int(o_t)
        if o_b - int(o_b) == 0.0:
            frame_list.append(frame)

    frame_list = list(set(frame_list))
    print(len(frame_list))
    svl_file.writelines([f'    <model id="7" name="" sampleRate="44100" start="{min(frame_list)}" end="{(max(frame_list)+1)}" type="sparse" dimensions="1" resolution="1" notifyOnAdd="true" dataset="6" />\n    <dataset id="6" dimensions="1">\n'])
    
    current_beat = -1
    for index,row in enumerate(data[1:]):

        o_t = int(row[0])
        o_b = float(row[4])
        frame = int(o_t)
        if o_b - int(o_b) == 0.0 and int(o_b) != current_beat:
            svl_file.writelines([f'\t<point frame="{frame}" label="{int(o_b)}_gt" />\n'])
            current_beat = int(o_b)

    svl_file.writelines(['    </dataset>\n  </data>\n  <display>\n    <layer id="8" type="timeinstants" name="Time Instants" model="7"  plotStyle="0" colourName="Green" colour="#008000" darkBackground="false"/>\n  </display>\n</sv>'])
    svl_file.close()