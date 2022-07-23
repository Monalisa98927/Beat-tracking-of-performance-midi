from __future__ import annotations
import os
from tkinter import FALSE
import numpy as np
import music21 as m21
import pretty_midi
import pandas as pd
import json
import csv
from pathlib import Path
from data_preprocessing_utils import *

import sys
sys.path.append('../..')

from ASAP.util import util
import importlib
from tqdm import tqdm
tqdm.pandas()

importlib.reload(util)


BASE_PATH = "../../ASAP/"
OUTPUT_PATH = 'asap_data/'
beat_lookup_table_path = 'asap_data/asap_beat_lookup_table.json'
# darklist = ['Ravel/Miroirs/4_Alborada_del_gracioso/CHOE02.mid', 'Schumann/Kreisleriana/2/JohannsonP03.mid', 'Scriabin/Sonatas/5/ChernovA06M.mid']


time_granularity = 100 # means the smallest time unit is 10ms
beat_granularity = 60 # 60 means 1/60 beat
pad_length = 400
sample_rate = 44100
beat_num_range = range(8,13)
length_range = [6, 400]


def read_midi_score_annotations(quant_df):

    for midi_score_annotations in quant_df['midi_score_annotations']:
        print(midi_score_annotations)
        with open(BASE_PATH + midi_score_annotations) as file:
            count_beat = 0
            for line in file:
                # print(line)
                count_beat += 1

            # print the number of beats of each midi score (235 midi score)
            print(count_beat)


def check_musicxml(quant_df):

    # check the musicxml
    quant_df.progress_apply(lambda row: util.xmlscore_parsable_music21(str(Path(BASE_PATH,row["xml_score"]))),axis = 1)
    for score_xml_path in quant_df['xml_score']:
        score = m21.converter.parse(BASE_PATH + score_xml_path)

def read_midi_score(quant_df):

    for midi_score in quant_df['midi_score']:

        midi_score = 'Scriabin/Sonatas/5/midi_score.mid'
        midi = read_midi_file(BASE_PATH + midi_score)

        print('\n--------------------------------------------')
        print(midi_score)
        
        instruments = midi.instruments    
        for i, instrument in enumerate(instruments):

            notes = instrument.notes
            program = instrument.program

            name = instrument.name
            name = name.replace(" ", "")

            print('\n', i, 'program', program, ' number of notes', len(notes))
            for note in notes:
                if note.start > 0.0 and note.end < 3.0:
                    print(f'{note.start:.3f}', f'{note.end:.3f}', note.pitch)

def read_performance_annotations(quant_df):

    for performance_annotations in quant_df['performance_annotations']:

        with open(BASE_PATH + performance_annotations) as file:
            for line in file:
                print(line)

def read_json_file(json_path):

    with open(json_path, 'r') as json_file:
        json_file = json.load(json_file)

    return json_file

def create_beat_lookup_table(json_file, quant_df):

    lookup_dic = {}

    for midi_performance in quant_df['midi_performance']:

        annotation = json_file[midi_performance]
        performance_beats_type = annotation['performance_beats_type']
        performance_time_signatures = annotation['perf_time_signatures']
        current_time_sig = 0

        lookup_dic[midi_performance] = {}

        # 给每首曲子的开头和结尾都加上1个假想的beats
        perf_0 = 0
        perf_1 = 0
        perf_2 = 0
        perf_3 = 0

        for i, perf in zip(range(len(performance_beats_type)), performance_beats_type):
            if i == 0:
                perf_0 = float(perf)
            elif i == 1:
                perf_1 = float(perf)
            elif i == len(performance_beats_type) - 2:
                perf_2 = float(perf)
            elif i == len(performance_beats_type) - 1:
                perf_3 = float(perf)

        # 计算第一个time signature，是为了处理第一个time signature出现之前的beat
        first_time_siganature = [performance_time_signatures[time_sig] for index, time_sig in zip(range(1), performance_time_signatures)][0][1]

        # 初始化beat phase positive/negative，首先需要计算每首曲子第一个downbeat出现的索引
        for i, perf in zip(range(len(performance_beats_type)), performance_beats_type):
            if performance_beats_type[perf] == 'db':
                first_downbeat_index = i + 1 # +1是因为在所有beats之前加了一个fake beat（index=0）
                break

        # before the first beat, add a fake beat
        beat_phase_pos = (0 - first_downbeat_index) % first_time_siganature
        beat_phase_neg = 0 if beat_phase_pos == 0 else (beat_phase_pos - first_time_siganature)
        fake_table_row_first = {
            'onset_time': perf_0 - (perf_1 - perf_0),
            'beat_type': 'db' if beat_phase_pos == 0 else 'b', 
            'beat_phase_pos': beat_phase_pos,
            'beat_phase_neg': beat_phase_neg,
            'time_sig': first_time_siganature
        }
        lookup_dic[midi_performance][0] = fake_table_row_first

        for i, perf in zip(range(len(performance_beats_type)), performance_beats_type):

            beat_index = i
            onset_time = perf
            perf_beat_type = performance_beats_type[perf]

            if onset_time in performance_time_signatures:
                current_time_sig = performance_time_signatures[onset_time][1]
            if current_time_sig == 0:
                # 如果第一个downbeat还未出现，则当前的time signature默认等于第一次出现的time signature
                current_time_sig = first_time_siganature

            # 从每一个downbeat开始计数，每次beat phase positive加1，直到再次遇到downbeat就再置零
            if perf_beat_type == 'db':
                beat_phase_pos = 0
                beat_phase_neg = 0
            else:
                beat_phase_pos = ((beat_index + 1) - first_downbeat_index) % int(current_time_sig)
                beat_phase_neg = ((beat_index + 1) - first_downbeat_index) % int(current_time_sig) - int(current_time_sig)
            
            table_row = {
                'onset_time': float(onset_time), 
                'beat_type': perf_beat_type, 
                'beat_phase_pos': beat_phase_pos,
                'beat_phase_neg': beat_phase_neg,
                'time_sig': int(current_time_sig)
            }

            # normal path
            lookup_dic[midi_performance][beat_index + 1] = table_row

        # after the last beat, add a fake beat
        beat_phase_pos = (beat_phase_pos + 1) % int(current_time_sig)
        beat_phase_neg = 0 if beat_phase_pos == 0 else (beat_phase_pos - int(current_time_sig))
        fake_table_row_last = {
            'onset_time': perf_3 + (perf_3 - perf_2),
            'beat_type': 'db' if beat_phase_pos == 0 else 'b', 
            'beat_phase_pos': beat_phase_pos,
            'beat_phase_neg': beat_phase_neg,
            'time_sig': int(current_time_sig)
        }
        lookup_dic[midi_performance][len(performance_beats_type) + 1] = fake_table_row_last

    return lookup_dic
                
def write_beat_lookup_table_to_json(quant_df):

    annotation_json_path = BASE_PATH + 'asap_annotations.json'
    json_file = read_json_file(annotation_json_path)
    lookup_dic = create_beat_lookup_table(json_file, quant_df)

    with open(beat_lookup_table_path, 'w') as write_f:
	    json.dump(lookup_dic, write_f, indent=4, ensure_ascii=False)

def get_beat_value(time, index, perf_beat_lookup_table, type):
    
    left_time = 0
    left_beat_phase_pos = 0
    left_beat_phase_neg = 0
    right_time = 0

    # when an onset is before all beats
    if index == -1:
        left_time = perf_beat_lookup_table["0"]['onset_time'] - \
            (perf_beat_lookup_table["1"]['onset_time'] - perf_beat_lookup_table["0"]['onset_time'])
        right_time = perf_beat_lookup_table["0"]['onset_time']

        # print(-1, type)
        return None, None, None, None

    # when an onset is after all beats
    elif index == -2:
        left_time = perf_beat_lookup_table[str(len(perf_beat_lookup_table) - 1)]['onset_time']
        right_time = perf_beat_lookup_table[str(len(perf_beat_lookup_table) - 1)]['onset_time'] + \
            (perf_beat_lookup_table[str(len(perf_beat_lookup_table) - 1)]['onset_time'] - perf_beat_lookup_table[str(len(perf_beat_lookup_table) - 2)]['onset_time'])

        # print(-2, type)
        return None, None, None, None

    else:
        left_time = perf_beat_lookup_table[str(index)]['onset_time']
        right_time = perf_beat_lookup_table[str(index + 1)]['onset_time']

        left_beat_phase_pos = perf_beat_lookup_table[str(index)]['beat_phase_pos']
        left_beat_phase_neg = perf_beat_lookup_table[str(index)]['beat_phase_neg']

    beat_subdiv = (time - left_time) / (right_time - left_time)

    # if index == -2:
    #     index = str(len(perf_beat_lookup_table) - 1)

    beat = float(index) + beat_subdiv

    return left_beat_phase_pos, left_beat_phase_neg, beat_subdiv, beat

def get_beat_for_all_note_events(quant_df, beat_lookup_table_path):

    train, validation, test = split_dataset(quant_df)
    beat_lookup_table = read_json_file(beat_lookup_table_path)

    for midi_performance in quant_df['midi_performance']:

        print(midi_performance)

        if midi_performance in train:
            split = 'train/'
        elif midi_performance in validation:
            split = 'validation/'
        elif midi_performance in test:
            split = 'test/'
        
        # annotation_path = OUTPUT_PATH + split + midi_performance.replace('/','-').replace('.mid', '.csv')
        annotation_path = BASE_PATH + midi_performance.replace('.mid', '.csv')

        file = open(annotation_path, 'w+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(['start_time', 'end_time', 'instrument', 'note', 'start_beat', 'duration_beat', 'beat_phase_pos', 'beat_phase_neg', 'beat_subdiv'])

        # get the corresponding beat from beat_lookup_table json file
        perf_beat_lookup_table = beat_lookup_table[midi_performance]

        midi = read_midi_file(BASE_PATH + midi_performance) 
        instruments = midi.instruments    
        for i, instrument in enumerate(instruments):

            notes = instrument.notes
            for note in notes:
                
                onset = float(note.start)
                offset = float(note.end)
                pitch = int(note.pitch)

                onset_left_index = -2
                offset_left_index = -2

                for index in perf_beat_lookup_table:

                    if perf_beat_lookup_table[index]['onset_time'] > onset:
                        onset_left_index = int(index) - 1
                        break

                for index in perf_beat_lookup_table:

                    if perf_beat_lookup_table[index]['onset_time'] > offset:
                        offset_left_index = int(index) - 1
                        break

                onset_beat_phase_pos, onset_beat_phase_neg, onset_subdiv, onset_beat = get_beat_value(onset, onset_left_index, perf_beat_lookup_table, 'onset')
                offset_beat_phase_pos, offset_beat_phase_neg, offset_subdiv, offset_beat = get_beat_value(offset, offset_left_index, perf_beat_lookup_table, 'offset')

                if onset_beat != None and offset_beat != None:
                    duration_beat = offset_beat - onset_beat

                    # format like musicnet .csv files
                    # start_time, end_time, instrument, note, start_beat, duration_beat
                    note_event_row = [onset, offset, 0, pitch, onset_beat, duration_beat, onset_beat_phase_pos, onset_beat_phase_neg, onset_subdiv]
                    assert duration_beat > 0
                    csv_writer.writerow(note_event_row)

        file.close()

def split_dataset(quant_df):

    train = []
    validation = []
    test = []

    for index, midi_perf in enumerate(quant_df['midi_performance']):
        if index % 10 < 8:
            train.append(midi_perf)
        elif index % 10 == 8:
            validation.append(midi_perf)
        elif index % 10 == 9:
            test.append(midi_perf)

    return train, validation, test

def calculate_nmat(quant_df, split='train'):

    train_set_ids, validation_set_ids, test_set_ids = split_dataset(quant_df)

    data_ids = []
    if split == 'train':
        data_ids = train_set_ids
    elif split == 'validation':
        data_ids = validation_set_ids
    elif split == 'test':
        data_ids = test_set_ids

    # initialize the nmat and nmat_length
    nmat_list = []
    nmat_length_list = []
    nmat_windows_list = []
    count_segments = 0 # count the number of the random beats segments

    # get the nmat and nmat_length of the training set
    for path in data_ids:
        file_path = BASE_PATH + path.replace('.mid', '.csv')
        nmat, nmat_length, num_of_segments, nmat_windows = \
            calculate_nmat_of_random_beats('asap', file_path, pad_length, beat_num_range, length_range, sample_rate, time_granularity, beat_granularity)

        count_segments += len(nmat_length)
        nmat_list += nmat
        nmat_length_list += nmat_length
        assert len(nmat)//pad_length == len(nmat_length)
        assert len(nmat_windows) == len(nmat_length)
    
    print(f'Number of input segments in {split} set:', count_segments)

    assert len(nmat_list)//pad_length == len(nmat_length_list)
    assert len(nmat_length_list) == len(nmat_windows_list)

    print(f'{split} set data size:', len(nmat_list), len(nmat_length_list))
    print(f'{split} set longest segment', max(nmat_length_list))
    print(f'{split} set avg segment length', sum(nmat_length_list) / len(nmat_length_list))

    np.save(f'asap_data/random_beats/{split}_nmat.npy', nmat_list)
    np.save(f'asap_data/random_beats/{split}_nmat_length.npy', nmat_length_list)

    return nmat_list, nmat_length_list, nmat_windows_list


def check_extremely_short_duration(path, data):

    # check the extremely short duration
    for i, note in enumerate(data[1:]):
        start_time = float(note[0]) # * 100 # in 10ms
        end_time = float(note[1]) # in 10ms
        start_beat = float(note[4])
        duration_beat = float(note[5])
        beat_subdiv = float(note[8])

        if round(100*(end_time - start_time)) < 1:
            print('duration time < 10ms', path, (end_time - start_time), duration_beat*60)
        
        if round(duration_beat*60) < 1:
            print('duration beat < 1/60 beat', path, (end_time - start_time), duration_beat*60)

def check_extremely_short_or_long_segments(path, perf_nmat_len_path):

    # check the extremely short and long segments
    with open(BASE_PATH + perf_nmat_len_path) as nmat_len_file:
        for i, line in enumerate(nmat_len_file):
            length = int(line.replace('\n',''))
            if length < 6:
                print(path, i, length)
            if length > 700:
                print(path, i, length)

def check_nmat(split):

    nmat = np.load(f'asap_data/random_beats/{split}_nmat.npy')
    nmat_len = np.load(f'asap_data/random_beats/{split}_nmat_length.npy')

    print(f'{split} set nmat length range:', min(nmat_len), max(nmat_len))
    print(f'{split} set o_t range:', min(nmat[:, 0]), max(nmat[:, 0]))
    print(f'{split} set d_t range:', min(nmat[:, 1]), max(nmat[:, 1]))
    print(f'{split} set pitch range:', min(nmat[:, 2]), max(nmat[:, 2]))
    print(f'{split} set o_b_pos range:', min(nmat[:, 3]), max(nmat[:, 3]))
    print(f'{split} set o_b_neg range:', min(nmat[:, 4]), max(nmat[:, 4]))
    print(f'{split} set o_b_subdiv range:', min(nmat[:, 5]), max(nmat[:, 5]))
    print(f'{split} set d_b range:', min(nmat[:, 6]), max(nmat[:, 6]))


if __name__ == '__main__':

    all_df = pd.read_csv(Path(BASE_PATH, "metadata.csv"))
    print(all_df.columns)

    quant_df = all_df.drop_duplicates(subset="performance_annotations", keep="first")
    print(quant_df)

    # write_beat_lookup_table_to_json(quant_df)
    # get_beat_for_all_note_events(quant_df, beat_lookup_table_path)

    # train_nmat, train_nmat_len = calculate_nmat(quant_df, 'train')
    # validation_nmat, validation_nmat_len = calculate_nmat(quant_df, 'validation')
    # test_nmat, test_nmat_len = calculate_nmat(quant_df, 'test')

    check_nmat('train')
    check_nmat('validation')
    check_nmat('test')
