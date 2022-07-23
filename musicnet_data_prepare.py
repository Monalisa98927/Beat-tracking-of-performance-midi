import csv
import os
from importlib_metadata import metadata
import numpy as np
from data_preprocessing_utils import *

###########################################

time_granularity = 100 # 100 means 10ms, 1000 means ms
beat_granularity = 60 # 60 means 1/60 beat
pad_length = 400 #350 #218
sample_rate = 44100
beat_num_range = range(8,13)
length_range = [6,350]

############################################

darklist = ['2292', '2194', '2227', '2230', '2305', '2211', '2310']
note_type_list = [
    'Unknown','Triplet Sixty Fourth', 'Sixty Fourth', \
    'Triplet Thirty Second', 'Thirty Second', 
    'Triplet Sixteenth', 'Sixteenth', 'Triplet', \
    'Dotted Sixteenth', 'Eighth', 'Dotted Eighth', 'Quarter', \
    'Tied Quarter-Thirty Second', 'Tied Quarter-Sixteenth', 'Dotted Quarter', \
    'Half', 'Dotted Half', 'Whole', 
]
note_type_set = []
note_type_ground_truth_dict = {
    'Unknown': 0,
    'Triplet Sixty Fourth': 1/24,
    'Sixty Fourth': 0.0625,
    'Triplet Thirty Second': 1/12,
    'Thirty Second': 0.125,
    'Triplet Sixteenth': 1/6,
    'Sixteenth': 0.25,
    'Triplet': 1/3,
    'Dotted Sixteenth': 0.375,
    'Eighth': 0.5,
    'Dotted Eighth': 0.75,
    'Quarter': 1,
    'Tied Quarter-Thirty Second': 1.125,
    'Tied Quarter-Sixteenth': 1.25,
    'Dotted Quarter': 1.5,
    'Half': 2,
    'Dotted Half': 3,
    'Whole': 4,
}
note_type_dic = {
    'Triplet Thirty Second': [],
    'Half': [],
    'Thirty Second': [],
    'Dotted Sixteenth': [],
    'Tied Quarter-Thirty Second': [],
    'Tied Quarter-Sixteenth': [],
    'Dotted Half': [],
    'Triplet': [],
    'Dotted Eighth': [],
    'Sixty Fourth': [],
    'Quarter': [],
    'Unknown': [],
    'Whole': [],
    'Triplet Sixty Fourth': [],
    'Triplet Sixteenth': [],
    'Eighth': [],
    'Dotted Quarter': [],
    'Sixteenth': []
}

# # music piece of 4/4 (all 72 pieces, 23894 seconds)
# ['1758', '1763', '1765', '1775', '1805', '1818', '1822', '1873', 
# '1916', '2075', '2076', '2078', '2079', '2080', '2081', '2112', 
# '2114', '2140', '2148', '2177', '2195', '2200', '2201', '2202', 
# '2208', '2212', '2213', '2214', '2215', '2218', '2221', '2224', 
# '2225', '2228', '2229', '2232', '2237', '2238', '2239', '2241', 
# '2242', '2247', '2283', '2284', '2288', '2293', '2294', '2297', 
# '2302', '2303', '2307', '2313', '2343', '2345', '2346', '2348', 
# '2350', '2377', '2383', '2388', '2391', '2417', '2436', '2441', 
# '2494', '2504', '2507', '2512', '2516', '2537', '2621', '2677']

# # solo_train_ids_4_4 (45):  
# ['1765', '2516', '2213', '2346', '2238', '1758', '2242', '2202', 
# '2215', '1763', '2239', '2677', '2218', '2288', '2343', '2208', 
# '2345', '2297', '2294', '2237', '2229', '2195', '2228', '2537', 
# '2200', '2388', '2512', '2348', '2391', '2441', '2302', '2214', 
# '2232', '2225', '2224', '2350', '2241', '2247', '2212', '1775', 
# '2436', '2221', '2293', '2201', '2307']
# # solo_test_ids_4_4 (1): 
# ['2303']

# # solo train data (26)
# ['1805', '1818', '1822', '1873', '1916', '2075', '2076', '2078', 
# '2079', '2080', '2081', '2112', '2114', '2140', '2148', '2177', 
# '2283', '2284', '2313', '2377', '2383', '2417', '2494', '2504', 
# '2507', '2621']

# can be rendered with piano: 
# 1805, 1818, 1822, 1873, 1916, 2075, 2076, 2078, 2079, 2080, 2081,
# 2112, 2114, 2140, 2148, ...

# Get the path of the MusicNet dataset
label_path = '/gpfsnyu/scratch/kf2395/musicnet/musicnet/musicnet'
midi_path = '/gpfsnyu/scratch/kf2395/musicnet/musicnet_midis/musicnet_midis'
metadata_path = '/gpfsnyu/scratch/kf2395/musicnet/musicnet_metadata.csv'

def get_musicnet_split(path, split):
    """
    Returns a list of all the files in the MusicNet dataset.
    """
    path = path + '/' + split + '_labels'
    ids = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            # only focus on .csv files, eliminate the _nmat.csv and _nmat_len.csv files
            if 'nmat' not in filename:
                ids.append(filename.split('.')[0])
    return ids

def get_musicnet_midi_files(path):
    """
    Returns a list of all the MIDI files in the MusicNet dataset.
    """
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def calculate_nmat_of_fixed_beats(file_path, beat_number=8.0):
    """
    Calculates note matrix of each song in the MusicNet dataset.
    将所有note events分成若干个8-beat(without overlap)
    """
    nmat = []
    nmat_length = []
    current_nmat_length = 0
    data = load_musicnet_csv(file_path)
    count_2_bar = 0

    if float(data[1][4]) > beat_number:
        count_2_bar = 1

    file_id = int(file_path.split('/')[-1].split('.')[0])
    # nmat.append([file_id,file_id,file_id,file_id,file_id])

    for index,row in enumerate(data[1:]):

        o_t = int(row[0])
        d_t = int(row[1]) - o_t
        instrument = row[2] # instrument information is not used
        pitch = int(row[3])
        d_b = float(row[5])
        note_type = row[6]

        if d_b not in note_type_dic[note_type]:
            note_type_dic[note_type].append(d_b)

        o_b = float(row[4])
        if count_2_bar == int(np.floor(o_b / beat_number)):
            
            nmat_row = [(o_t*time_granularity)//sample_rate, (d_t*time_granularity)//sample_rate, pitch, o_b % beat_number, d_b]
            nmat.append(nmat_row)
            current_nmat_length += 1
        else:
            
            if not current_nmat_length == 0:
                nmat_length.append(current_nmat_length)

                # pad nmat to pad_length
                while current_nmat_length < pad_length:
                    current_nmat_length += 1
                    nmat_row = [0, 0, 0, 0, 0]
                    nmat.append(nmat_row)

                count_2_bar = int(np.floor(o_b / beat_number))
                current_nmat_length  = 0

    if not current_nmat_length == 0:
        nmat_length.append(current_nmat_length)

        # pad nmat to pad_length
        while current_nmat_length < pad_length:
            current_nmat_length += 1
            nmat_row = [0, 0, 0, 0, 0]
            nmat.append(nmat_row)

    if len(nmat) == 0 and len(nmat_length) == 0:
        print(file_path)
        print(len(nmat),len(nmat_length))

    return nmat, nmat_length, len(nmat_length)


def calculate_nmat_of_fixed_notes(file_path, beat_num_range):
    """
    Calculates note matrix of each song in the MusicNet dataset.
    每100个note events为一个input,重叠率30%

    input: file path of a music piece
    """
    nmat = []
    nmat_length = []
    data = load_musicnet_csv(file_path)

    left = 0 # unit (note)
    right = 100 # unit (note)
    step_size = 0.3 * right
    pad_length = 100
    while len(data)-1-1 >= left:

        current_nmat_length = 0

        for index,row in enumerate(data[1:]):
            o_t = int(row[0]) * time_granularity // sample_rate
            d_t = int(row[1]) * time_granularity // sample_rate - o_t
            pitch = int(row[3])
            o_b = float(row[4])
            d_b = float(row[5])

            if index >= left and index < right:
                nmat_row = [o_t, d_t, pitch, o_b, d_b]
                nmat.append(nmat_row)
                current_nmat_length += 1

        if current_nmat_length != 0:
            nmat_length.append(current_nmat_length)

            # pad nmat to pad_length
            while current_nmat_length < pad_length:
                current_nmat_length += 1
                nmat_row = [0, 0, 0, 0, 0]
                nmat.append(nmat_row)

        left += step_size
        right += step_size

    return nmat, nmat_length, len(nmat_length)


def get_musicnet_solo(metadata_path):

    metadata = load_musicnet_metadata(metadata_path)

    solo_ids = []
    # time_sum = 0

    for item in metadata[1:]:
        if 'Solo' in item[4]:
            solo_ids.append(item[0])
            # time_sum += int(item[8])
    
    return solo_ids

def load_musicnet_metadata(csv_file):
    """
    Loads a CSV file containing the MusicNet metadata.
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_audio_duration(id):
    
    metadata = load_musicnet_metadata(metadata_path)

    for item in metadata[1:]:
        if item[0] == id:
            return int(item[8])

def get_all_4_4_segments(id, midi):
    """
    Returns a list of all 4/4 segments in a MIDI file.
    """
    
    cut_dur = 0
    start_time_list_of_4_4 = []
    end_time_list_of_4_4 = []
    time_sig_change = midi.time_signature_changes
    audio_duration = get_audio_duration(id)

    for t in time_sig_change:
      # print(t)
      if t.numerator == 4 and t.denominator == 4:
        cut_dur = t.time
        start_time_list_of_4_4.append(t.time)
      else:
        if len(start_time_list_of_4_4) > len(end_time_list_of_4_4):
          end_time_list_of_4_4.append(t.time)

    if len(start_time_list_of_4_4) == len(end_time_list_of_4_4) + 1:
      end_time_list_of_4_4.append(audio_duration)

    time_range_of_4_4 = []
    for start,end in zip(start_time_list_of_4_4,end_time_list_of_4_4):
      time_range_of_4_4.append([start, end])

    return time_range_of_4_4

def get_4_4_music_ids(midi_files):
    """
    Returns a list of ids with 4/4 time signature.
    """
    audio_duration_of_4_4 = 0
    music_list_of_4_4 = []

    # Get the ids with 4/4 time signature
    for midi_file in midi_files:
        midi_id = midi_file.split('/')[-1].split('.')[0].split('_')[0]

        if midi_id not in darklist: # and midi_id in solo_ids:
            midi_data = read_midi_file(midi_file)
            time_sig_changes = midi_data.time_signature_changes

            # select the ids with time_sig_changes = [4/4]
            if len(time_sig_changes) == 1 \
                and time_sig_changes[0].numerator == 4 \
                and time_sig_changes[0].denominator == 4:
                audio_duration_of_4_4 += get_audio_duration(midi_id)
                music_list_of_4_4.append(midi_id)

    print('Audio duration of 4/4 music pieces:', audio_duration_of_4_4)
    print('Number of 4/4 music pieces:', len(music_list_of_4_4))

    return music_list_of_4_4

def absolute_onset_time_to_relative_onset_time(nmat):
    """
    Converts absolute onset time to relative onset time.
    """

    current_start_time = 0
    nmat_ = []
    for i, nmat_row in enumerate(nmat):
        if i % pad_length == 0:
            current_start_time = nmat[i][0]
        if not (nmat[i][0] == 0 and nmat[i][1] == 0):
            nmat[i][0] -= current_start_time
        nmat_.append(nmat[i])

    return nmat_

def absolute_onset_beat_to_relative_onset_beat(nmat):
    """
    Converts absolute onset beat to relative onset beat.
    """

    current_start_beat = 0
    nmat_ = []
    for i, nmat_row in enumerate(nmat):
        if i % pad_length == 0:
            current_start_beat = nmat[i][3]
        if not (nmat[i][0] == 0 and nmat[i][1] == 0):
            nmat[i][3] -= current_start_beat
        nmat_.append(nmat[i])
    
    return nmat_

def float_beat_value_to_int(nmat, multiple):
    """
    Converts float onset_beat and duration_beat to int onset_beat and duration_beat.
    """
    err_onset = 0.
    err_duration = 0.
    for i, nmat_row in enumerate(nmat):
        if (nmat_row[3] != 0 or nmat_row[4] != 0):

            nmat_row[3] *= multiple
            nmat_row[4] *= multiple

            err_onset += abs(nmat_row[3] - round(nmat_row[3]))
            err_duration += abs(nmat_row[4] - round(nmat_row[4]))

            # if abs(nmat_row[3] - round(nmat_row[3])) > 0.3 or abs(nmat_row[4] - round(nmat_row[4])) > 0.3:
            #     print(nmat_row[3], nmat_row[4])
            
            nmat_row[3] = round(nmat_row[3])
            nmat_row[4] = round(nmat_row[4])

    # print('Multiple:', multiple, 'Error in onset:', err_onset)
    # print('Multiple:', multiple, 'Error in duration:', err_duration)

    return nmat


def check_nmat(split, type=''):

    nmat = np.load(f'musicnet_nmat{type}/{split}_nmat_list.npy')
    nmat_len = np.load(f'musicnet_nmat{type}/{split}_nmat_length_list.npy')
    nmat_win = np.load(f'musicnet_nmat{type}/{split}_nmat_windows_list.npy')

    new_nmat = []

    count = 0

    for i, note_len in enumerate(nmat_len):
        # print('\n', 'sequence length =', note_len)
        segment = nmat[i*pad_length: i*pad_length + note_len]

    print('\n')
    print(f'{split} set segement number:', len(nmat_len))
    print(f'{split} set nmat length range:', min(nmat_len), max(nmat_len))
    print(f'{split} set o_t range:', min(nmat[:, 0]), max(nmat[:, 0]))
    print(f'{split} set d_t range:', min(nmat[:, 1]), max(nmat[:, 1]))
    print(f'{split} set pitch range:', min(nmat[:, 2]), max(nmat[:, 2]))
    print(f'{split} set o_b range:', min(nmat[:, 3]), max(nmat[:, 3]))
    print(f'{split} set d_b range:', min(nmat[:, 4]), max(nmat[:, 4]))
    
    print(len(nmat), len(nmat_len))
    assert len(nmat) / pad_length == len(nmat_len)


def calculate_nmat(data_ids, split='train'):

    # initialize the nmat and nmat_length
    nmat_list = []
    nmat_length_list = []
    nmat_windows_list = []
    count_segments = 0 # count the number of the random beats segments

    folder = 'test' if split == 'test' else 'train' 

    # get the nmat and nmat_length of the training set
    for i in data_ids:
        file_path = label_path + f'/{folder}_labels/' + i + '.csv'
        nmat, nmat_length, num_of_segments, nmat_windows = \
            calculate_nmat_of_random_beats('musicnet', file_path, pad_length, beat_num_range, length_range, sample_rate, time_granularity, beat_granularity)

        count_segments += len(nmat_length)
        nmat_list += nmat
        nmat_length_list += nmat_length
        nmat_windows_list += nmat_windows
        assert len(nmat)//pad_length == len(nmat_length)
        assert len(nmat_windows) == len(nmat_length)

    print(f'Number of input segments in {split} set:', count_segments)

    assert len(nmat_list)//pad_length == len(nmat_length_list)
    assert len(nmat_length_list) == len(nmat_windows_list)

    print(f'{split} set data size:', len(nmat_list), len(nmat_length_list))
    print(f'{split} set longest segment', max(nmat_length_list))
    print(f'{split} set avg segment length', sum(nmat_length_list) / len(nmat_length_list))

    np.save(f'musicnet_nmat/random_beats/{split}_nmat.npy', nmat_list)
    np.save(f'musicnet_nmat/random_beats/{split}_nmat_length.npy', nmat_length_list)
    np.save(f'musicnet_nmat/random_beats/{split}_nmat_windows.npy', nmat_windows_list)

    return nmat_list, nmat_length_list, nmat_windows_list
    
if __name__ == '__main__':
    """
    Main function.
    """
    # Get the ids of train/test set in the MusicNet dataset
    train_set_ids = get_musicnet_split(label_path, 'train')
    train_set_ids = [id for id in train_set_ids if id not in darklist]
    test_set_ids = get_musicnet_split(label_path, 'test')
    test_set_ids = [id for id in test_set_ids if id not in darklist] # 10 pieces

    # split training set into train/validation set
    validation_set_ids = train_set_ids[int(len(train_set_ids)*0.8):]
    train_set_ids = train_set_ids[:int(len(train_set_ids)*0.8)]

    calculate_nmat(train_set_ids, 'train')
    calculate_nmat(validation_set_ids, 'validation')
    calculate_nmat(test_set_ids, 'test')

    # check_nmat('train', '/fixed_8_beats')
    # check_nmat('validation', '/fixed_8_beats')
    # check_nmat('test', '/fixed_8_beats')


######################################################################

    # # Get all midi files in the MusicNet dataset
    # midi_files = get_musicnet_midi_files(midi_path)

    # # Get the solo ids in the MusicNet dataset
    # solo_ids = get_musicnet_solo(metadata_path)
    # print('Number of solo music:', len(solo_ids))

    # # Get the ids with 4/4 time signature in the dataset
    # music_list_of_4_4 = get_4_4_music_ids(midi_files)

    # # initialize the nmat and nmat_length
    # train_nmat_list = []
    # train_nmat_length_list = []
    # validation_nmat_list = []
    # validation_nmat_length_list = []
    # test_nmat_list = []
    # test_nmat_length_list = []
    # count_2_bar = 0 # count the number of 2-bar in all 4/4 music pieces

    # # split the training set and test set of solo ids with 4/4 time signature
    # train_ids_4_4 = list(set(music_list_of_4_4) & set(train_set_ids))
    # validation_ids_4_4 = list(set(music_list_of_4_4) & set(validation_set_ids))
    # print('Train 4/4:', len(train_ids_4_4))
    # print('Validation 4/4:', len(validation_set_ids))

    # # get the nmat and nmat_length of the training set
    # for i in train_set_ids:
    #     file_path = label_path + '/train_labels/' + i + '.csv'
    #     nmat, nmat_length, num_of_2_bar = calculate_nmat_of_random_beats(file_path, beat_num_range)

    #     count_2_bar += len(nmat_length)
    #     train_nmat_list += nmat
    #     train_nmat_length_list += nmat_length
    #     assert len(nmat)//pad_length == len(nmat_length)

    # # # calculate the onset_time.txt
    # # onset_time_list = []
    # # for item in test_nmat_list:
    # #     if not sum(item) == 0:
    # #         onset_time = item[0]
    # #         # print(onset_time)
    # #         onset_time_list.append(str(onset_time)+'\n')
    # # onset_time_file = open('onset_time.txt',mode='w')
    # # onset_time_file.writelines(onset_time_list)
    # # onset_time_file.close()

    # # # convert absolute onset time to relative onset time
    # # train_nmat_list = absolute_onset_time_to_relative_onset_time(train_nmat_list)
    # # validation_nmat_list = absolute_onset_time_to_relative_onset_time(validation_nmat_list)
    # # test_nmat_list = absolute_onset_time_to_relative_onset_time(test_nmat_list)

    # # # convert absolute onset beat to relative onset beat
    # # train_nmat_list = absolute_onset_beat_to_relative_onset_beat(train_nmat_list)
    # # validation_nmat_list = absolute_onset_beat_to_relative_onset_beat(validation_nmat_list)
    # # test_nmat_list = absolute_onset_beat_to_relative_onset_beat(test_nmat_list)

    # # # convert float onset_beat and duration_beat to int onset_beat and duration_beat
    # # train_nmat_list = float_beat_value_to_int(train_nmat_list, beat_granularity)
    # # validation_nmat_list = float_beat_value_to_int(validation_nmat_list, beat_granularity)
    # # test_nmat_list = float_beat_value_to_int(test_nmat_list, beat_granularity)

    # # 48216 35178 104 3838 49430
    # # 0 0 0 0 0
    # # 4821 3517 104 480 3570
    # # 49 10 10; 36 10 10; 6? 2 12; 8 60; 8 8 60